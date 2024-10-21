module TBMFT

    export TightBindingMFT, GetMFTEnergy

    using TightBindingToolkit, LinearAlgebra, Logging

    using ..MeanFieldToolkit.MFTBonds: GetBondCoorelation


@doc """
`TightBindingMFT{T, R}` is a data type representing a general mean-field simulation on a tight-binding model.

# Attributes
- `model              ::  Model`: The tight-binding model on which mean-field simulations are going to run, contains info about free hoppings.
- `HoppingOrders      ::  Vector{Param{2, R}}`: a vector of order parameters to decompose the interactions in during MFT.
- `Interactions       ::  Vector{Param{2, FLoat64}}`: the vector of `Param` containing all the information of the interactions acting on the model.
- `MFTDecomposition   ::  Vector{Function}` : the decomposition function which describes how to take an interaction array + expectation values and give back tight-binding hoppings.
- `MFTScaling         ::  Dict{String, Float64}`: relative scaling parameters for different mean-field channels.
- `ChannelLabels      ::  Dict{String, String}`: The labels of the different mean-field channels.

Initialize this structure using
```julia
TightBindingMFT(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}}, MFTDecomposition::Vector{Function} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
TightBindingMFT(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}}, MFTDecomposition::Vector{Function}, MFTScaling::Dict{String, Float64} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site"))
```
"""
    mutable struct TightBindingMFT{T, R}
        ##### TightBinding Model for MFT
        model               ::  Model
        ##### Vector of Params containing the expected order parameters of MFT
        HoppingOrders       ::  Vector{Param{2, R}}
        ##### Vector of Params tracking information about the Interacting UnitCell, and the corresponding MFT decomposition functions
        Interactions        ::  Vector{Param{T, Float64}}
        MFTDecomposition    ::  Vector{Function}
        ##### The MFT expectation value of the full interacting Hamiltonian
        MFTEnergy           ::  Vector{Float64}
        ##### The relative scaling b/w different MFT channels
        MFTScaling          ::  Dict{String, Any}
        ##### The user defined labels of the different MFT channels
        ChannelLabels       ::  Dict{String, String}


        function TightBindingMFT(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}} , MFTDecomposition::Vector{Function} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            @warn "`MFTScaling` attribute not passed. Resorting to default values of uniform relative scaling for every channel!"
            MFTScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            return new{T, R}(model, HoppingOrders, Interactions, MFTDecomposition, Float64[], MFTScaling, ChannelLabels)
        end

        function TightBindingMFT(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}}, MFTDecomposition::Vector{Function}, MFTScaling::Dict{String, Any} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            return new{T, R}(model, HoppingOrders, Interactions, MFTDecomposition, Float64[], MFTScaling, ChannelLabels)
        end

        function TightBindingMFT(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}} , MFTDecomposition::Function ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            @warn "`MFTScaling` attribute not passed. Resorting to default values of uniform relative scaling for every channel!"
            MFTScaling      =   Dict{String, Float64}("ij" => 1.0, "ii" => 1.0, "jj" => 1.0)

            return new{T, R}(model, HoppingOrders, Interactions, repeat(Function[MFTDecomposition], length(Interactions)), Float64[], MFTScaling, ChannelLabels)
        end

        function TightBindingMFT(model::Model, HoppingOrders::Vector{Param{2, R}}, Interactions::Vector{Param{T, Float64}}, MFTDecomposition::Function, MFTScaling::Dict{String, Any} ; ChannelLabels :: Dict{String, String} = Dict{String, String}("ij" => "Hopping", "ii" => "Hopping On-Site", "jj" => "Hopping On-Site")) where {T, R <: Union{Float64, ComplexF64}}

            return new{T, R}(model, HoppingOrders, Interactions, repeat(Function[MFTDecomposition], length(Interactions)), Float64[], MFTScaling, ChannelLabels)
        end

    end


@doc """
```julia
GetMFTEnergy(tbMFT::TightBindingMFT{T, R}) --> Float64
```
Returns the total mean-field energy of the model including decomposed interactions.

"""
    function GetMFTEnergy(tbMFT::TightBindingMFT{T, R}) :: Float64 where {T, R}
        ##### /// TODO: Add Free Hopping energies also
        ##### /// TODO: Test!!!!

        Energy      =   0.0
        lookup      =   Lookup(tbMFT.model.uc)

        for BondKey in keys(lookup)

            G_ij        =   GetBondCoorelation(tbMFT.model.Gr, BondKey..., tbMFT.model.uc, tbMFT.model.bz)

            t_ij        =   lookup[BondKey]
            Energy      +=  sum((t_ij .* G_ij))
        end

        return real(Energy) / length(tbMFT.model.uc.basis)
    end










end
