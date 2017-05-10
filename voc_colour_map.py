"""
Script to generate color palette for PASCAL VOC
"""
def get_bit(num, idx):
    if ( num & 2**idx > 0 ):
        return 1
    else:
        return 0

def voc_colour_map(N = 256):

    palette = []

    for i in range(N):
        idx= i
        r = 0
        g = 0
        b = 0

        for j in range(8):
            r_bit = get_bit(idx, 0)
            g_bit = get_bit(idx, 1)
            b_bit = get_bit(idx, 2)
            mul_factor = 2**(7-j)

            r_mask = r_bit * mul_factor
            g_mask = g_bit * mul_factor
            b_mask = b_bit * mul_factor

            r = r | r_mask
            b = b | b_mask
            g = g | g_mask

            idx= idx// 8

        palette.append(r)
        palette.append(g)
        palette.append(b)

    return palette

