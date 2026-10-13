"""Rewrite a Linux dump into the Windows byte form: every glibc shortest-form %a
token (the golden dump's x= labels, every value of the C3 FP32 dump) becomes MSVC's
fixed 13-digit %a form, and lines end in CRLF (MSVC text-mode stdout).

Usage: python msvc_norm.py <linux dump> > <windows-form dump>"""
import re, struct, sys
def msvc_a(tok):
    v = float.fromhex(tok)
    bits = struct.unpack('>Q', struct.pack('>d', v))[0]
    sign = '-' if bits >> 63 else ''
    e = (bits >> 52) & 0x7ff; m = bits & ((1 << 52) - 1)
    if e == 0:
        lead, exp = 0, (-1022 if m else 0)
    else:
        lead, exp = 1, e - 1023
    return f"{sign}0x{lead}.{m:013x}p{exp:+d}"
data = open(sys.argv[1], encoding='ascii').read()
data = re.sub(r'-?0x[0-9a-f.]+p[+-]\d+', lambda mo: msvc_a(mo.group(0)), data)
sys.stdout.buffer.write(data.replace('\n', '\r\n').encode('ascii'))
