module MFTRun
    export SolveMFT!

    using FixedPointToolkit, Distributions, TightBindingToolkit, Logging, LinearAlgebra

    using ..MeanFieldToolkit.TBMFT: TightBindingMFT
    using ..MeanFieldToolkit.BDGMFT: BdGMFT
    using ..MeanFieldToolkit.MFTIter: MFTIterator


@doc """
```julia
SolveMFT!(mft::TightBindingMFT{T, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) --> SelfCons
SolveMFT!(mft::BdGMFT{T, R, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) --> SelfCons 
SolveMFT!(mft::TightBindingMFT{T, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) --> SelfCons 
SolveMFT!(mft::BdGMFT{T, R, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) --> SelfCons 
SolveMFT!(mft::TightBindingMFT{T, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) --> SelfCons
SolveMFT!(mft::BdGMFT{T, R, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) --> SelfCons 
SolveMFT!(mft::TightBindingMFT{T, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) --> SelfCons
SolveMFT!(mft::BdGMFT{T, R, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) --> SelfCons
```
Solves the mean-field theory on the given `MFT` object, and returns the `SelfCons` object (Refer to [FixedPointToolkit](https://github.com/Anjishnubose/FixedPointToolkit.jl)) containing the results of the mean-field theory.
- If `fileName` is passed, then the `SelfCons` object is saved to the file after every `checkpoint_interval` iterations.
- If `Initial` is passed, then the initial order parameters are set to the values in `Initial`.
- If `Initial_range` is passed, then the initial order parameters are set to random values in the range `Initial_range`.
- If `Update` is passed, then the update function is used to perform the self-consistency update.
- If `Update_kwargs` is passed, then the keyword arguments are passed to the update function.
- If `max_iter` is passed, then the maximum number of iterations is set to `max_iter`.
- If `tol` is passed, then the tolerance for convergence is set to `tol`.

"""
    function SolveMFT!(mft::TightBindingMFT{T, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingOrders)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT{T, R, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingOrders) + length(mft.PairingOrders)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end


    function SolveMFT!(mft::TightBindingMFT{T, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingOrders)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT{T, R, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingOrders) + length(mft.PairingOrders)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end


    function SolveMFT!(mft::TightBindingMFT{T, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT{T, R, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::TightBindingMFT{T, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT{T, R, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        GetGap!(mft.model)

        convergence     =   norm(selfcons.VOuts[end] - selfcons.VIns[end]) / sqrt(length(selfcons.VOuts[end]))
        @info "COMPLETED with convergence = $(convergence)!"
        return selfcons
    end


end