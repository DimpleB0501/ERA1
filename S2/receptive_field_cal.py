import math
# in max pool layer stride is 2, kernel size = 2 and padding = 0
s = [1, 1, 2, 1, 1, 2, 1, 1, 1]
p = [1, 1, 0, 1, 1, 0, 0, 0, 0]

n_in = 28
r_in = 1
j_in = 1
k = 3

print ('n_in', '\t', 'r_in', '\t', 'j_in', '\t', 's', '\t', 'p', '\t', 'n_out', '\t', 'r_out', '\t', 'j_out')
for r in range(len(s)):
    if s[r] == 2:
        k = 2
    else:
        k = 3
    n_out = math.floor(((n_in + (2*p[r]) - k) / s[r]) + 1)
    r_out = r_in + ((k-1)*j_in)
    j_out = j_in * s[r]

    print (n_in, '\t', r_in, '\t', j_in, '\t', s[r], '\t', p[r], '\t', n_out, '\t', r_out, '\t', j_out)

    n_in = n_out
    r_in = r_out
    j_in = j_out
