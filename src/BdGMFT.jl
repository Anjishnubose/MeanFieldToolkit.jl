module BDGMFT

    export BdGMFT, GetMFTEnergy

    using TightBindingToolkit, LinearAlgebra, Logging

    using ..MeanFieldToolkit.MFTBonds: GetBondDictionary
    using ..MeanFieldToolkit.Blocks: ParamBlock

    import ..MeanFieldToolkit.TBMFT: GetMFTEnergy
    mutable struct BdGMFT{T} 
        ##### The full BdG MFT Model, and the two blocks tracking hopping and pairing expectation values and order parameters.
        bdgModel                ::  BdGModel
        HoppingBlock            ::  ParamBlock{2}
        PairingBlock            ::  ParamBlock{2}
        ##### The ParamBlock tracking information about the Interacting UnitCell, and the corresponding MFT decomposition functions for hopping and pairing
        InteractionBlock        ::  Vector{ParamBlock{T}}
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


        function BdGMFT(bdgModel::BdGModel, HoppingBlock::ParamBlock{2}, PairingBlock::ParamBlock{2}, InteractionBlock::Vector{ParamBlock{T}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T}

            @warn "Scaling attributes not passed. Resorting to default values of uniform relative scaling for every channel!"
            HoppingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)
            PairingScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            @assert IsSameUnitCell( bdgModel.uc_hop, HoppingBlock.uc) "Inconsistency between Tight-Binding Unit Cell and hopping Expectation Unit Cell"
            @assert IsSameUnitCell(bdgModel.uc_pair, PairingBlock.uc) "Inconsistency between Pairing Unit Cell and pairing Expectation Unit Cell"

            return new{T}(bdgModel, HoppingBlock, PairingBlock, InteractionBlock, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFT(bdgModel::BdGModel, HoppingBlock::ParamBlock{2}, PairingBlock::ParamBlock{2}, InteractionBlock::Vector{ParamBlock{T}} , HoppingDecomposition::Vector{Function}, PairingDecomposition::Vector{Function}, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T}

            @assert isSameUnitCell( bdgModel.uc_hop, HoppingUC) "Inconsistency between Tight-Binding Unit Cell and hopping Expectation Unit Cell"
            @assert isSameUnitCell(bdgModel.uc_pair, PairingUC) "Inconsistency between Pairing Unit Cell and pairing Expectation Unit Cell"

            return new{T}(bdgModel, HoppingBlock, PairingBlock, InteractionBlock, HoppingDecomposition, PairingDecomposition, Float64[], HoppingScaling, PairingScaling, HoppingLabels, PairingLabels)

        end

        function BdGMFT(bdgModel::BdGModel, HoppingBlock::ParamBlock{2}, PairingBlock::ParamBlock{2}, InteractionBlock::ParamBlock{T} , HoppingDecomposition::Function, PairingDecomposition::Function ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T}

            return BdGMFT(bdgModel, HoppingBlock, PairingBlock, ParamBlock{T}[InteractionBlock], Function[HoppingDecomposition], Function[PairingDecomposition] ; HoppingLabels = HoppingLabels, PairingLabels = PairingLabels)

        end

        function BdGMFT(bdgModel::BdGModel, HoppingBlock::ParamBlock{2}, PairingBlock::ParamBlock{2}, InteractionBlock::ParamBlock{T} , HoppingDecomposition::Function, PairingDecomposition::Function, HoppingScaling::Dict{String, Float64}, PairingScaling::Dict{String, Float64} ; HoppingLabels::Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"), PairingLabels::Dict{String, String} = Dict{String, String}("ij" => "Pairing", "ii" => "Pairing On-Site", "jj" => "Pairing On-Site")) where {T}

            return BdGMFT(bdgModel, HoppingBlock, PairingBlock, ParamBlock{T}[InteractionBlock], Function[HoppingDecomposition], Function[PairingDecomposition], HoppingScaling, PairingScaling ; HoppingLabels = HoppingLabels, PairingLabels = PairingLabels)

        end

    end

    ##### /// TODO: Add Free Hopping and pairing energies also
    ##### TODO: Test
    function GetMFTEnergy(bdgMFT::BdGMFT{T}) :: Float64 where {T}

        Energy             =   0.0
        HoppingLookup      =   Lookup(bdgMFT.bdgModel.uc_hop)
        PairingLookup      =   Lookup(bdgMFT.bdgModel.uc_pair)

        for BondKey in keys(HoppingLookup)
            
            Expectations        =   GetBondDictionary(bdgMFT.HoppingBlock.lookup, BondKey, bdgMFT.HoppingBlock.uc.localDim)

            Chi_ij              =   get(Expectations, "ij", zeros(ComplexF64, repeat([bdgMFT.HoppingBlock.uc.localDim], 2)...))
            t_ij                =   HoppingLookup[BondKey]

            Energy              +=  sum((t_ij .* Chi_ij))
        end

        for BondKey in keys(PairingLookup)
            
            Expectations        =   GetBondDictionary(bdgMFT.PairingBlock.lookup, BondKey, bdgMFT.PairingBlock.uc.localDim)

            Delta_ij            =   get(Expectations, "ij", zeros(ComplexF64, repeat([bdgMFT.PairingBlock.uc.localDim], 2)...))
            p_ij                =   PairingLookup[BondKey]

            Energy              +=  sum((p_ij .* Delta_ij))
        end

        return real(Energy) / length(bdgMFT.HoppingBlock.uc.basis)
    end










end