Sampling references captured from commit edbaf8c with the released atomweaver.pt,
CPU PyTorch 2.14.0, two threads, and seed 123 immediately before sampling.
The input is the first three peptide residues and a small target fragment from
examples/9RA5_MK8.pdb. Each case uses three reverse steps and production shell
jitter scaling (0.25), five element classes, and reserved-slot-zero exemption.

The cases cover joint design, subset design with residue 2 pinned to the input,
and two neighbor recycle passes. Inputs and all returned outputs (including
coordinate trajectories) are included. These short CPU checks detect numerical
regressions; they do not replace full-length GPU validation.
