"""Rule-based ISCN karyotype parser: free-text karyotype -> 0/1 flags (structured karyotype representation of the mortality task).

Per clone: modal number, sex chromosomes, abnormality tokens (clone references and constitutional variants are not counted).
Flags (0 when the karyotype is not available; avail = 0): avail, normal, complex (>= 3 abnormalities), monosomal (Breems 2008),
m7_del7q, m5_del5q, t9_22, cbf_apl, kmt2a, inv3, t6_9, abn17p, tri8. `parse(text)` returns a dict with these flags plus counts.
"""
import re

NA_PAT = re.compile(r"not\s*done|no\s*data|no\s*mitos|fail|no\s*metaphase|insufficient|unknown|n/?a\b", re.I)
REF = re.compile(r"^(idem|sl|sdl\d*|ids?|inc|cp)$", re.I)          # clone references / "incomplete" marker
MODAL = re.compile(r"^\d{2,3}([~-]\d{2,3})?(<\d?n>)?$", re.I)
SEX = re.compile(r"^[XY]{1,6}\??$", re.I)
NUMTOK = re.compile(r"^[+-](\d{1,2}|X|Y)(x\d)?c?\??$", re.I)
MARTOK = re.compile(r"^[+-]?\d*([~-]\d+)?(mar|dmin|r)\d*(x\d)?\??$", re.I)
STRUCT_NAMES = "t|del|dup|inv|add|der|i|ins|idic|dic|r|ider|hsr|trp|tan|rob|psu|fra|tas|qdp|ss|hsr"
STRUCT = re.compile(r"^[+-]?\??((" + STRUCT_NAMES + r")\([^()]*\)(\([^()]*\))*)+(x\d)?c?\??$", re.I)
POLY_INV9 = re.compile(r"^inv\(9\)\(p1\d(\.\d+)?q[12]\d?(\.\d+)?\)c?$", re.I)
T922 = re.compile(r"t\(9[;:]22\)", re.I)
TRANS = re.compile(r"t\(([0-9XY]{1,2})[;:]([0-9XY]{1,2})(?:[;:]([0-9XY]{1,2}))?\)\(([^;:)]+)[;:]([^;:)]+)(?:[;:]([^)]+))?\)", re.I)


def split_top(c):
    """split on commas that are not inside parentheses (some breakpoints are written t(9;22)(q34,q11))"""
    parts, depth, cur = [], 0, ""
    for ch in c:
        if ch == "(": depth += 1
        elif ch == ")": depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p]


def split_clones(s):
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(\d{2,3})\.([XY])", r"\1,\2", s)       # typo: dot instead of comma after the modal number
    out = []
    for c in re.split(r"/{1,2}", s):
        c = re.sub(r"\[[^\]]*\]", "", c)          # clone size [20], [cp10]
        parts = split_top(c)
        if parts:
            out.append(parts)
    return out


def parse(s):
    """Return dict(status, n_abn, flags..., n_unrec). status in {'missing','not_available','parsed','partial'}."""
    keys = ["avail", "normal", "complex", "monosomal", "m7_del7q", "m5_del5q", "t9_22", "cbf_apl", "kmt2a", "inv3",
            "t6_9", "abn17p", "tri8"]
    res = {k: 0 for k in keys}
    res.update(n_abn=0, n_unrec=0, n_tokens=0)
    if s is None or (isinstance(s, float)) or str(s).strip() == "":
        res["status"] = "missing"; return res
    s = str(s)
    if NA_PAT.search(s) and not re.search(r"\d{2},", s):
        res["status"] = "not_available"; return res
    clones = split_clones(s)
    abn = set(); modal_ok = True; poly = False; mono_auto = set(); struct_n = 0; unrec = 0; ntok = 0
    for parts in clones:
        rest = parts
        if MODAL.match(parts[0]):
            mn = int(re.match(r"\d+", parts[0]).group())
            if mn != 46: modal_ok = False
            if mn >= 69: poly = True
            rest = parts[1:]
            if rest and SEX.match(rest[0]):
                sx = rest[0].upper().rstrip("?")
                if mn < 69 and len(sx) == 1:            # 45,X or 45,X,... : a sex chromosome is missing
                    abn.add("sexloss")
                rest = rest[1:]
        for p in rest:
            ntok += 1
            if REF.match(p): continue
            if re.match(r"^X[XY]$", p, re.I): continue          # sex field written out of place (no modal number before it)
            if POLY_INV9.match(p) or (p.endswith("c") and STRUCT.match(p)): continue
            if NUMTOK.match(p) or MARTOK.match(p):
                abn.add(p.lower().rstrip("?"))
                m = re.match(r"^-(\d{1,2})", p)
                if m: mono_auto.add(int(m.group(1)))
            elif STRUCT.match(p):
                abn.add(p.lower().rstrip("?")); struct_n += 1
            elif NA_PAT.search(p):
                continue
            else:
                unrec += 1; abn.add("unrec:" + p.lower())
            pl = p.lower()
            if re.match(r"^-7(?!\d)", pl) or re.match(r"^del\(7\)\(q", pl): res["m7_del7q"] = 1
            if re.match(r"^-5(?!\d)", pl) or re.match(r"^del\(5\)\(q", pl): res["m5_del5q"] = 1
            if re.match(r"^-17(?!\d)", pl) or re.match(r"^(del|add)\(17\)\(p", pl) or re.match(r"^i\(17\)\(q10\)", pl): res["abn17p"] = 1
            if re.match(r"^\+8(?!\d)", pl): res["tri8"] = 1
            if re.search(r"inv\(16\)|t\(16;16\)", pl) or re.search(r"t\(8[;:]21\)", pl) or re.search(r"t\(15[;:]17\)", pl): res["cbf_apl"] = 1
            if re.search(r"inv\(3\)\(q2\d", pl) or re.search(r"t\(3[;:]3\)", pl): res["inv3"] = 1
            if re.search(r"t\(6[;:]9\)", pl): res["t6_9"] = 1
            if T922.search(pl): res["t9_22"] = 1
            for m in TRANS.finditer(pl):
                chs = [m.group(1), m.group(2), m.group(3)]; bands = [m.group(4), m.group(5), m.group(6)]
                for c, b in zip(chs, bands):
                    if c == "11" and b and b.startswith("q23"): res["kmt2a"] = 1
    if poly: abn.add("polyploid")
    n = len(abn)
    res.update(avail=1, n_abn=n, n_unrec=unrec, n_tokens=ntok,
               normal=int(n == 0 and modal_ok), complex=int(n >= 3),
               monosomal=int(len(mono_auto) >= 2 or (len(mono_auto) >= 1 and struct_n >= 1)),
               status=("partial" if unrec else "parsed"))
    return res
