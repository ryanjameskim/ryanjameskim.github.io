# MultiLinear Expansion

As part of the Zero Knowledge Hackathon, I've been learning about the mechanism behind ZK proofs. Justin Thaler's book [https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf](Proofs, Arguments, and Zero-Knowledge) is a rigorous introduction to the material and has some great exercises that are worth doing to solidify understanding.

Here is a python script that I wrote as part of the book's exercises. For any function that maps n-dimensions of binary numbers {0,1} to an integer, one can derive a unique multi-linear function which shares the original inputs --> outputs, but also expands the domain of each dimension. In the example below for instance, f(0,1) = 4 in the original function. Likewise, g(0,1) = 4, but as you can see the domain areas have expanded from binary to a p=5 field. The exact equation is g(x1,x2) = 3 - 2(x_1) + 1(x_2) and it is the ONLY solution which has at most 1 degree for each x_i (no powers, but cross terms like x_1 * x_2 allowed).

**ORIGINAL FUNCTION f(x1,x2)**
   |  #0 |  #1 |

#0 |   3 |   4 |

#1 |   1 |   2 |

**Multi Linear Expansion g(x1,x2)**
   |  #0 |  #1 |  #2 |  #3 |  #4 |

#0 |   3 |   4 |   0 |   1 |   2 |

#1 |   1 |   2 |   3 |   4 |   0 |

#2 |   4 |   0 |   1 |   2 |   3 |

#3 |   2 |   3 |   4 |   0 |   1 |

#4 |   0 |   1 |   2 |   3 |   4 |

This process is called multilinear expansion and is used in zero knowledge proofs to expand the collision area in which errors of an incorrect input become magnified and much easier to catch for a verifier.

The script calculates the multilinear expansion using two different methods of various speeds.

Something extra that the exercise did not call for but I found useful to do is the script uses a tree-like multiplication method to calculate the individual coefficients for each term to write out the equation nicely.

https://github.com/ryanjameskim/public/blob/master/multinomialexpansion_final.py
