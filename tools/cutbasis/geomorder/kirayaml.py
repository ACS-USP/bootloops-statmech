# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of BootLoops 1.0; see LICENSE and NOTICE beside this file.
"""Read a Kira config dir (integralfamilies.yaml + kinematics.yaml) into a geomorder family spec.

Also the command-line argument checks shared by cutbasis.py, geo_master_select.py and evaluate.py: a UsageError names
the offending option in one line and the drivers exit with code 2 (argparse's usage-error code). family_header and
parse_sectors read only the YAML, so a driver can refuse a bad option before importing sympy."""
from __future__ import annotations
import os, yaml

USAGE_RC = 2


class UsageError(ValueError):
    """A command-line option the tool cannot act on; the message is the one named line."""


def family_header(config_dir, family=None):
    """(family name, number of propagators, names of all families) from integralfamilies.yaml; UsageError if the
    config directory or the family is not there."""
    path = os.path.join(config_dir, 'integralfamilies.yaml')
    if not os.path.isfile(path):
        raise UsageError(f'--config {config_dir}: no integralfamilies.yaml there')
    with open(path) as f:
        fams = yaml.safe_load(f)['integralfamilies']
    names = [str(x['name']) for x in fams]
    if family is None:
        famd = fams[0]
    else:
        hits = [x for x in fams if str(x['name']) == family]
        if not hits:
            raise UsageError(f"--family {family}: not in {path} (families: {', '.join(names)})")
        famd = hits[0]
    return str(famd['name']), len(famd['propagators']), names


def parse_sectors(text, top, nprops, family):
    """'--sectors a,b,c' -> [a, b, c]; every entry must be an integer sector number of the family and a subsector of
    --top (UsageError otherwise). Also checks --top itself."""
    hi = (1 << nprops) - 1
    if not 1 <= top <= hi:
        raise UsageError(f'--top {top}: sector numbers of the {nprops}-propagator family {family} run from 1 to {hi}')
    out = []
    for tok in text.split(','):
        try:
            s = int(tok)
        except ValueError:
            raise UsageError(f'--sectors {text}: {tok!r} is not an integer sector number') from None
        if not 1 <= s <= hi:
            raise UsageError(f'--sectors {text}: sector {s} is not one of the {nprops}-propagator family {family} (1 to {hi})')
        if s | top != top:
            raise UsageError(f'--sectors {text}: sector {s} is not a subsector of --top {top}')
        out.append(s)
    return out


def read_kira_family(config_dir, family=None, d0=4, scale=None):
    import sympy as sp
    fam_name, _, _ = family_header(config_dir, family)
    with open(os.path.join(config_dir, 'integralfamilies.yaml')) as f:
        fams = yaml.safe_load(f)['integralfamilies']
    with open(os.path.join(config_dir, 'kinematics.yaml')) as f:
        kin = yaml.safe_load(f)['kinematics']
    famd = next(x for x in fams if str(x['name']) == fam_name)
    loops = [str(x) for x in famd['loop_momenta']]
    incoming = [str(x) for x in kin.get('incoming_momenta', [])]
    outgoing = [str(x) for x in kin.get('outgoing_momenta', [])]
    allext = incoming + outgoing
    momsyms = {m: sp.Symbol(m) for m in loops + allext}
    cons = kin.get('momentum_conservation', None)
    consub = {}
    if cons:
        dep = str(cons[0]); expr = sp.sympify(str(cons[1]), locals=momsyms)
        consub[momsyms[dep]] = expr
    ext = [m for m in allext if momsyms[m] not in consub]
    invariants = [str(x[0]) for x in kin.get('kinematic_invariants', [])]
    one = kin.get('symbol_to_replace_by_one', None)
    kinsyms = {s: sp.Symbol(s) for s in invariants}
    repl1 = {kinsyms[str(one)]: 1} if one is not None and str(one) in kinsyms else {}
    symbols = [s for s in invariants if not (one is not None and s == str(one))]
    sp_rules = {}
    for (pair, val) in kin.get('scalarproduct_rules', []):
        a, b = str(pair[0]), str(pair[1])
        v = sp.sympify(str(val), locals=kinsyms).subs(repl1)
        sp_rules[(a, b)] = v
    # express rules for the independent ext momenta only; dependent momenta (p4) are expanded via conservation.
    def dot(a, b):
        ea = consub.get(momsyms[a], momsyms[a]); eb = consub.get(momsyms[b], momsyms[b])
        prod = sp.expand(ea * eb)
        # replace products of independent momenta symbols by rules
        res = sp.Integer(0)
        for term in sp.Add.make_args(prod):
            c, rest = term.as_coeff_Mul()
            syms = []
            for fct in sp.Mul.make_args(rest):
                if fct.is_Pow:
                    syms += [str(fct.base)] * int(fct.exp)
                else:
                    syms.append(str(fct))
            assert len(syms) == 2, term
            key = (syms[0], syms[1]) if (syms[0], syms[1]) in sp_rules else (syms[1], syms[0])
            if key not in sp_rules:
                raise ValueError(f'no scalar product rule for {key}')
            res += c * sp_rules[key]
        return sp.expand(res)
    rules = {}
    for i, a in enumerate(ext):
        for b in ext[i:]:
            rules[f'{a}*{b}'] = str(dot(a, b))
    props = []
    for pr in famd['propagators']:
        pstr = str(pr[0])
        if '^2' in pstr or '**2' in pstr:
            # generic quadratic form "(q)^2 - m2" : q linear in momenta
            e = sp.sympify(pstr.replace('^', '**'), locals={**momsyms, **kinsyms})
            e = sp.expand(e.subs(consub))
            # find q: take the part quadratic in momenta; assume (sum c_i p_i)^2 + const
            momlist = [momsyms[m] for m in loops + ext]
            quad = sum(t for t in sp.Add.make_args(e) if any(t.has(m) for m in momlist))
            const = sp.expand(e - quad)
            # recover q from the coefficient structure: q = sum c_m m with c_m^2 = coeff of m^2
            q = 0
            ref = None
            for m in momlist:
                c2 = quad.coeff(m, 2)
                if c2 != 0:
                    if ref is None:
                        ref = m; cref = sp.sqrt(c2); q += cref * m
                    else:
                        # sign from cross term with ref
                        cross = quad.coeff(ref, 1).coeff(m, 1)
                        q += (cross / (2 * cref)) * m
            if sp.expand(q ** 2 - quad) != 0:
                raise ValueError(f'cannot parse quadratic propagator {pstr}')
            momexpr = sp.expand(q)
            m2 = sp.expand(-const).subs(repl1) + sp.sympify(str(pr[1]), locals=kinsyms).subs(repl1)
        else:
            momexpr = sp.sympify(pstr, locals=momsyms).subs(consub)
            momexpr = sp.expand(momexpr)
            m2 = sp.sympify(str(pr[1]), locals=kinsyms).subs(repl1)
        mom = {}
        for m in loops + ext:
            c = momexpr.coeff(momsyms[m])
            if c != 0:
                mom[m] = str(c)
        props.append({'mom': {k: (int(sp.nsimplify(v)) if sp.nsimplify(v).is_integer else str(v)) for k, v in mom.items()}, 'm2': str(m2)})
    if scale is None:
        # pick a scale: the symbol replaced by one (=1) -> use 1; else first invariant
        scale = '1'
    spec = {'name': famd['name'], 'loops': loops, 'ext': ext, 'symbols': symbols, 'd0': d0, 'sigma_sign': 1,
            'scale': str(scale), 'sp_rules': rules, 'props': props,
            'top_level_sectors': famd.get('top_level_sectors', []), 'kira_config_dir': os.path.abspath(config_dir)}
    return spec


def sector_props(sector_id, nprops):
    """Kira sector number (binary, prop 1 = bit 0) -> sorted 1-based prop list"""
    return [i + 1 for i in range(nprops) if (sector_id >> i) & 1]


def sector_id(props):
    return sum(1 << (i - 1) for i in props)
