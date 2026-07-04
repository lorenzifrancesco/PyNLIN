# Blanking Condition for the Tiny Triangle Islands

In one unit cell, write

```math
x = i + u, \qquad y = j + v, \qquad u,v \in [-1/2, 1/2].
```

The small upper-right triangle is the part of the cell where

```math
u + v > 1/2.
```

Blanking a half-width `f` around the three transition lines gives the surviving
conditions

```math
u \le 1/2 - f,
```

```math
v \le 1/2 - f,
```

```math
u + v \ge 1/2 + f.
```

Such a point exists only if the largest allowed value of `u + v` still reaches
the lower diagonal cutoff:

```math
(1/2 - f) + (1/2 - f) \ge 1/2 + f.
```

Therefore

```math
f \le 1/6.
```

The tiny triangle collapses to a point at

```math
f = 1/6,
```

and disappears completely for

```math
f > 1/6.
```

## Relation to Channel Spacing and Bandwidth

If frequencies are normalized by the channel spacing `Delta f_ch`, and each
channel has occupied bandwidth `B`, then the normalized blank half-width around
the midpoint between adjacent channels is

```math
f = \frac{\Delta f_{\rm ch} - B}{2\Delta f_{\rm ch}}
  = \frac{1}{2}\left(1 - \frac{B}{\Delta f_{\rm ch}}\right).
```

The triangle disappears when

```math
\frac{1}{2}\left(1 - \frac{B}{\Delta f_{\rm ch}}\right) > \frac{1}{6},
```

or equivalently

```math
\frac{B}{\Delta f_{\rm ch}} < \frac{2}{3}.
```

Thus the condition for disappearance is

```math
B < \frac{2}{3}\Delta f_{\rm ch}.
```

At equality, `B = 2 Delta f_ch / 3`, the island has zero area.

## Numerical Verification

The plotting code was exercised directly through
`plot_blanked_round_regions_fixed.py::blanked_round` on the central tiny island
with triplet `(round(x+y), round(x), round(y)) = (1, 0, 0)`.

Using a `3001 x 3001` grid over one normalized cell,

```text
f threshold = 1/6 = 0.16666666666666666

cutoff f        surviving grid points in triplet (1,0,0)
0.050000000                   551920
0.100000000                   179101
0.150000000                    11371
0.165666667                       45
0.166666667                        0
0.167666667                        0
0.180000000                        0
0.200000000                        0
0.250000000                        0
```

The count goes to zero at `f = 1/6` because the surviving island has collapsed
to a zero-area point. For any `f > 1/6`, the island is absent.
