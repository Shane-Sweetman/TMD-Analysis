# Observable Definition

The hard process considered in the simulation is

```text
e+ e- -> Z/gamma* -> q qbar
```

at the Z pole, `sqrt(s) = 91.2 GeV`.

The event thrust axis is reconstructed from visible final-state particles and used as the event-level reference direction. Jets are reconstructed with FastJet anti-kT clustering with radius parameter `R = 0.4`.

For the selected charged pion pair in opposite jets, define

```text
q = p_pi1 + p_pi2
qT = |q - (q . n_thrust) n_thrust|
```

where `n_thrust` is the unit thrust-axis vector.

The pair is classified as:

- opposite-sign, if the two selected pion charges multiply to a negative value
- same-sign, if the two selected pion charges multiply to a positive value

The analysis studies qT spectra for several pion momentum-fraction cuts. The fraction used in the main PYTHIA code is approximately

```text
z_like = |p_pion| / |p_jet|
```

and the cut is applied to the lower of the two selected pion fractions. The thesis-level study used cuts including 0%, 20%, 40%, and 60%.
