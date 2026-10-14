# Bundled IPOPT binaries: sources and licenses

This folder contains prebuilt Linux x86-64 binaries of the IPOPT nonlinear optimization solver and
the shared libraries it depends on. They are included so that the TRAD agent runs without a separate
solver installation. None of this code was written by the TRAD authors; it is redistributed
unmodified under the licenses listed below.

**Source.** The files were copied from the conda-forge `ipopt` package (Ipopt 3.14.19, built with the
MUMPS linear solver) and its dependencies, as installed by `conda install -c conda-forge ipopt`.
Recipe: https://github.com/conda-forge/ipopt-feedstock. The only change is that `lib/libblas.so.3`,
a symbolic link to `lib/liblapack.so.3` in the conda package, is stored as a regular copy of the same
file.

The license texts are in `licenses/`. Licenses are as declared by the conda-forge packages; the
texts were taken from the current conda-forge builds of the same projects.

| Files | Project | License | Upstream |
|---|---|---|---|
| `bin/ipopt`, `lib/libipopt.so.3`, `lib/libipoptamplinterface.so.3` | Ipopt (COIN-OR) | EPL-2.0 (`licenses/IPOPT-LICENSE`) | https://github.com/coin-or/Ipopt |
| `lib/libasl.so` | AMPL Solver Library | BSD-3-Clause and SMLNJ | https://github.com/ampl/asl |
| `lib/libdmumps_seq.so`, `lib/libmumps_common_seq.so`, `lib/libpord_seq.so`, `lib/libmpiseq_seq.so` | MUMPS (sequential) | CeCILL-C | https://mumps-solver.org |
| `lib/libscotch.so.7.0`, `lib/libscotcherr.so.7.0`, `lib/libesmumps.so.7.0` | SCOTCH | CeCILL-C | https://gitlab.inria.fr/scotch/scotch |
| `lib/libmetis.so` | METIS | Apache-2.0 | https://github.com/KarypisLab/METIS |
| `lib/libspral.so` | SPRAL | BSD-3-Clause | https://github.com/ralna/spral |
| `lib/libblas.so.3`, `lib/liblapack.so.3` | OpenBLAS (provides BLAS and LAPACK) | BSD-3-Clause | https://github.com/OpenMathLib/OpenBLAS |
| `lib/libhwloc.so.15` | hwloc | BSD-3-Clause | https://www.open-mpi.org/projects/hwloc/ |
| `lib/libxml2.so.16` | libxml2 | MIT | https://gitlab.gnome.org/GNOME/libxml2 |
| `lib/libicuuc.so.78`, `lib/libicudata.so.78` | ICU | Unicode License V3 | https://icu.unicode.org |
| `lib/libiconv.so.2` | GNU libiconv | LGPL-2.1 | https://www.gnu.org/software/libiconv/ |
| `lib/liblzma.so.5` | XZ Utils (liblzma) | 0BSD | https://tukaani.org/xz/ |
| `lib/libbz2.so.1.0` | bzip2 | bzip2 license | https://sourceware.org/bzip2/ |
| `lib/libz.so.1` | zlib | Zlib | https://zlib.net |
| `lib/libgcc_s.so.1`, `lib/libgfortran.so.5`, `lib/libquadmath.so.0`, `lib/libgomp.so.1`, `lib/libstdc++.so.6` | GCC runtime libraries | GPL-3.0 with the GCC Runtime Library Exception 3.1 | https://gcc.gnu.org |

Source code for every component is available from the upstream links above and from the
corresponding conda-forge feedstocks (https://github.com/conda-forge).

To use your own IPOPT instead, delete this folder and install it with
`conda install -c conda-forge ipopt`; the agent then uses the `ipopt` found on the `PATH`.
