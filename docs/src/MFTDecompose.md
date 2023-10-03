# MeanFieldToolkit.MFTDecompose

```@autodocs
Modules = [MeanFieldToolkit, MeanFieldToolkit.MFTDecompose]
Private = false
Pages   = ["MFTDecompose.jl"]

```

# How to write custom mean-field equations

* Suppose you are dealing with a 2n-fermion interation on two sites of the general form $U^{ij}_{\alpha_1...\alpha_n ; \beta_1...\beta_n}f^{\dagger}_{i, \alpha_1}f_{i, \alpha_2}...f^{\dagger}_{j, \beta_{n-1}}f_{j, \beta_{n}}$. All the information about the interaction will be stored in the rank = 2n arrays $U^{ij}$.  
* Furthermore, you will need the expectation value matrices $\chi_{rr'}^{\alpha\beta} = \langle f^{\dagger}_{r, \alpha}f_{r', \beta}\rangle$, for $\chi_{ii},\chi_{jj},$ and $\chi_{ij}$. Assume this is passed as a dictionary of matrices with the keys being ``"ii", "jj", and "ij"``.
* Now, the function which decomposes the interaction on the bond connecting sites ``i, j`` can be defined as
```julia
    function CustomDecomposition(U::Array{Float64, 2*n}, Expectations::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}

        ##### Write down your mean field equations and find their on-site component t_{ii}, t_{jj}, and inter-site t_{ij}

        return Dict("ii" => t_ii, "jj" => t_jj, "ij" => t_ij)
    end
```

