module InteractionConvert
    export SpinToPartonCoupling, DensityToPartonCoupling

    using Tullio, LinearAlgebra, TightBindingToolkit

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


    function DensityToPartonCoupling(Ui::Matrix{Float64}, Uj::Matrix{Float64}) :: Array{ComplexF64, 4}
        @tullio U[a, b, c, d] := Ui[a, b] * Uj[c, d]
        return U
    end








end