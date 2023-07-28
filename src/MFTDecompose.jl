module MFTDecompose
    export InterQuarticToHopping, InterQuarticToPairing, IntraQuarticToHopping, IntraQuarticToPairing
    
    using TightBindingToolkit, LinearAlgebra, Tullio


    function InterQuarticToHopping(Uij::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}
        
        Chi_ij  =   Chis["ij"]
        Chi_ii  =   Chis["ii"] ##### Had to unpack Chis in this manner because Tullio doesnt deal well with dictionaries.
        Chi_jj  =   Chis["jj"]

        @tullio t_ij[ a , b]    :=   - conj(Chi_ij[c, d]) * Uij[a, c, d, b]
        @tullio t_ii[ a , b]    :=   Chi_jj[c, d] * Uij[a, b, c, d]
        @tullio t_jj[ a , b]    :=   Chi_ii[c, d] * Uij[c, d, a, b]

        return Dict("ij" => t_ij, 
                    "ii" => t_ii,
                    "jj" => t_jj)
    end


    function InterQuarticToPairing(Uij::Array{ComplexF64, 4}, Deltas::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}
        
        Delta_ij    =  Deltas["ij"]

        @tullio p_ij[ a , b]    :=    conj(Delta_ij[c, d]) * Uij[a, c, b, d]

        return Dict("ij" => p_ij)
    end


    function IntraQuarticToHopping(Uii::Array{ComplexF64, 4}, Chis::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}

        Chi_ii =    Chis["ii"] 

        @tullio t_ii[ a , b]    :=   Chi_ii[c, d] * (Uii[a, b, c, d] + Uii[c, d, a, b] - Uii[c, b, a, d] - Uii[a, d, c, b])

        return Dict("ii" => t_ii)
    end


    function IntraQuarticToPairing(Uii::Array{ComplexF64, 4}, Deltas::Dict{String, Matrix{ComplexF64}}) :: Dict{String, Matrix{ComplexF64}}

        Delta_ii =    Deltas["ii"] 

        @tullio p_ii[ a , b]    :=   conj(Delta_ii[c, d]) * (Uii[a, c, b, d])

        return Dict("ii" => p_ii)
    end




end