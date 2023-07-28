module MFTBonds
    export GetMFTBonds, GetBondDictionary

    using TightBindingToolkit, LinearAlgebra, Tullio

    function GetBondDictionary(BondLookup::Dict{Tuple, Matrix{ComplexF64}}, BondKey::Tuple{Int64, Int64, Vector{Int64}}, localDim::Int64) :: Dict{String, Matrix{ComplexF64}}
        base, target, offset    =   BondKey
        AdjBondKey              =   (target, base, -offset)
        
        if BondKey != AdjBondKey
            Expectation_ij      =   (         get(BondLookup,    BondKey, zeros(ComplexF64, repeat([localDim], 2)...)) 
                                    + adjoint(get(BondLookup, AdjBondKey, zeros(ComplexF64, repeat([localDim], 2)...))))

        else
            Expectation_ij      =   (         get(BondLookup,    BondKey, zeros(ComplexF64, repeat([localDim], 2)...)) 
                                    + adjoint(get(BondLookup, AdjBondKey, zeros(ComplexF64, repeat([localDim], 2)...)))) / 2
        end
        
        Expectation_ii      =   get(BondLookup, (  base,   base, zeros(Int64, length(offset))), zeros(ComplexF64, repeat([localDim], 2)...))
        Expectation_jj      =   get(BondLookup, (target, target, zeros(Int64, length(offset))), zeros(ComplexF64, repeat([localDim], 2)...))

        Expectations        =   Dict{String, Matrix{ComplexF64}}("ij" => Expectation_ij, "ii" => Expectation_ii, "jj" => Expectation_jj)

        return Expectations

    end

    
    function GetMFTBonds(DecomposedBonds::Dict{String, Matrix{ComplexF64}} ; BondKey::Tuple{Int64, Int64, Vector{Int64}}, uc::UnitCell{2}, scaling::Dict{String, Float64} = Dict("ij" => 1.0, "ii" => 1.0, "jj" => 1.0), labels::Dict{String, String} = Dict("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) :: Vector{Bond{2}} 
        
        base, target, offset = BondKey

        t_ij        =   get(DecomposedBonds, "ij", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        t_ii        =   get(DecomposedBonds, "ii", zeros(ComplexF64, repeat([uc.localDim], 2)...))
        t_jj        =   get(DecomposedBonds, "jj", zeros(ComplexF64, repeat([uc.localDim], 2)...))

        t_ijBond    =   Bond(BondKey..., scaling["ij"] * t_ij, GetDistance(uc, BondKey...), "MFT => $(labels["ij"]) : $(string(BondKey))") 

        t_iiBond    =   Bond(  base,   base, zeros(Int64, length(offset)), scaling["ii"] * t_ii, 0.0, "MFT => $(labels["ii"]) : $(string(BondKey))")
        t_jjBond    =   Bond(target, target, zeros(Int64, length(offset)), scaling["jj"] * t_jj, 0.0, "MFT => $(labels["jj"]) : $(string(BondKey))")

        return [t_ijBond, t_iiBond, t_jjBond]

    end



end