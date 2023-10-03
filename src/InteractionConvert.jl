module InteractionConvert
    export SpinToPartonCoupling, DensityToPartonCoupling

    using Tullio, LinearAlgebra, TightBindingToolkit


@doc """
```julia
SpinToPartonCoupling(J::Matrix{Float64}, spin::Rational) --> Array{ComplexF64, 4}
SpinToPartonCoupling(J::Matrix{Float64}, SpinVec_A::Vector{Matrix{ComplexF64}}, SpinVec_B::Vector{Matrix{ComplexF64}}) --> Array{ComplexF64, 4}

```
Converts the exchange matrix `J` into a parton coupling array `U` using the spin matrices `SpinVec_A` and `SpinVec_B`.
If `spin` is passed, then the spin matrices are generated using the `SpinMats` function.
"""
    function SpinToPartonCoupling(J::Matrix{Float64}, spin::Rational) :: Array{ComplexF64, 4}

        SpinVec         =   SpinMats(spin)[1: end - 1]
        @assert length(SpinVec) == size(J)[begin] "Inconsistent exchange matrix and spin given!"
        @tullio U[a, b, c, d] := J[i, j] * SpinVec[i][a, b] * SpinVec[j][c, d]

        return U
    end

    function SpinToPartonCoupling(J::Matrix{Float64}, SpinVec_A::Vector{Matrix{ComplexF64}}, SpinVec_B::Vector{Matrix{ComplexF64}}) :: Array{ComplexF64, 4}

        @assert length(SpinVec_A) == length(SpinVec_B) == size(J)[begin] "Inconsistent exchange matrix and spins given!"
        @tullio U[a, b, c, d] := J[i, j] * SpinVec_A[i][a, b] * SpinVec_B[j][c, d]

        return U
    end


@doc """
```julia
DensityToPartonCoupling(Ui::Matrix{Float64}, Uj::Matrix{Float64}) --> Array{ComplexF64, 4}
```
Converts the density-density interaction matrix `Ui` and `Uj` into a parton coupling array `U`.
"""
    function DensityToPartonCoupling(Ui::Matrix{Float64}, Uj::Matrix{Float64}) :: Array{ComplexF64, 4}
        @tullio U[a, b, c, d] := Ui[a, b] * Uj[c, d]
        return U
    end








end