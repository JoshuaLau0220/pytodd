# pytodd

An simple implementation of the TODD algorithm for reducing the T count of a quantum circuit in the paper
[[1712.01557] An Efficient Quantum Compiler that reduces $T$ count](https://arxiv.org/abs/1712.01557)
by Luke E Heyfron and Earl T Campbell (2019).

The TODD algorithm is a method for reducing the T count of a quantum circuit. It is based on an extension to the Lempel's algorithm for the 2-STR problem.

Currently only support inputing the phase polynomial via a binary matrix in the csv format. The output is a reduced phase polynomial in the csv format

## Usage

```bash
python3 pytodd.py <input.csv> <output.csv> [-v]
```
