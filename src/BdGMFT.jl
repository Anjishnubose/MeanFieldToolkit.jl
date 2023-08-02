module BDGMFT

    export BdGMFT, GetMFTEnergy

    using TightBindingToolkit, LinearAlgebra, Logging

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation
    using ..MeanFieldToolkit.Blocks: ParamBlock

    import ..MeanFieldToolkit.TBMFT: GetMFTEnergy
    mutable struct BdGMFT{T, R, S} 
        ##### The full BdG MFT Model, and the two blocks tracking hopping and pairing expectation values and order parameters.
        bdgModel                ::  BdGModel
        HoppingBlock            ::  ParamBlock{2, R}
        PairingBlock            ::  ParamBlock{2, S}
        ##### The ParamBlock tracking information about the Interacting UnitCell, and the corresponding MFT decomposition functions for hopping and pairing
        InteractionBlock        ::  Vector{ParamBlock{T, Float64}}
        HoppingDecomposition    ::  Vector{Function}
        PairingDecomposition    ::  Vector{Function}
        ##### The MFT expectation value of the full interacting Hamiltonian
        MFTEnergy               ::  Vector{Float64}
        ##### The relative scaling b/w different MFT channels for hopping and pairing
        HoppingScaling          ::  Dict{String, Float64}
        PairingScaling          ::  Dict{String, Float64}
        ##### The user defined labels of the different MFT channels for hopping and pairing
        HoppingLabels           ::  Dict{String, String}
        PairingLabels           ::  Dict{String, String}


        function BdGMFT(bdgModel::BdGModel, HoppingBlock::ParamBlock{2, R}, PairingBlock::ParamBlock{2, S}, InteractionBlock::Vector{ParamBlock{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            @warn "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            @assert IsSameUnitCell( bdgModel.uc_hop, HoppingBlock.uc) "Inconsistency between Tight-Binding Unit Cell and hopping Expectation Unit Cell"
            @assert IsSameUnitCell(bdgModel.uc_pair, PairingBlock.uc) "Inconsistency between Pairing Unit Cell and pairing Expectation Unit Cell"

            return new{T, R, S}(bdgModel, HoppingBlock, PairingBlock, InteractionBlock, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFT(bdgModel::BdGModel, HoppingBlock::ParamBlock{2, R}, PairingBlock::ParamBlock{2, S}, InteractionBlock::Vector{ParamBlock{T, Float64}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function}, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            @assert isSameUnitCell( bdgModel.uc_hop, HoppingUC) "Inconsistency between Tight-Binding Unit Cell and hopping Expectation Unit Cell"
            @assert isSameUnitCell(bdgModel.uc_pair, PairingUC) "Inconsistency between Pairing Unit Cell and pairing Expectation Unit Cell"

            return new{T, R, S}(bdgModel, HoppingBlock, PairingBlock, InteractionBlock, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFT(bdgModel::BdGModel, HoppingBlock::ParamBlock{2, R}, PairingBlock::ParamBlock{2, S}, InteractionBlock::ParamBlock{T, Float64} , HoppingDecomposition::Function, PairingDecomposition::Function ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            return BdGMFT(bdgModel, HoppingBlock, PairingBlock, ParamBlock{T, Float64}[InteractionBlock], Function[HoppingDecomposition], Function[PairingDecomposition] ; HoppingLabels = HoppingLabels, PairingLabels = PairingLabels)

        end

        function BdGMFT(bdgModel::BdGModel, HoppingBlock::ParamBlock{2, R}, PairingBlock::ParamBlock{2, S}, InteractionBlock::ParamBlock{T, Float64} , HoppingDecomposition::Function, PairingDecomposition::Function, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T, R <: Union{Float64, ComplexF64}, S <: Union{Float64, ComplexF64}}

            return BdGMFT(bdgModel, HoppingBlock, PairingBlock, ParamBlock{T, Float64}[InteractionBlock], Function[HoppingDecomposition], Function[PairingDecomposition], HoppingScaling, PairingScaling ; HoppingLabels = HoppingLabels, PairingLabels = PairingLabels)

        end

    end

    ##### /// TODO: Add Free Hopping and pairing energies also
    ##### TODO: Test
    function GetMFTEnergy(bdgMFT::BdGMFT{T, R, S}) :: Float64 where {T, R, S}

        Energy             =   0.0
        HoppingLookup      =   Lookup(bdgMFT.bdgModel.uc_hop)
        PairingLookup      =   Lookup(bdgMFT.bdgModel.uc_pair)

        for BondKey in keys(HoppingLookup)

            G_ij        =   GetBondCoorelation(bdgMFT.bdgModel.Gr, BondKey..., bdgMFT.bdgModel.uc_hop, bdgMFT.bdgModel.bz)
            t_ij        =   HoppingLookup[BondKey]

            Energy      +=  sum((t_ij .* G_ij))
        end

        for BondKey in keys(PairingLookup)

            F_ij        =   GetBondCoorelation(bdgMFT.bdgModel.Fr, BondKey..., bdgMFT.bdgModel.uc_pair, bdgMFT.bdgModel.bz)
            p_ij        =   PairingLookup[BondKey]

            Energy      +=  sum((p_ij .* F_ij))
        end

        return real(Energy) / length(bdgMFT.HoppingBlock.uc.basis)
    end










end