module MFTRun
    export SolveMFT!

    using FixedPointToolkit, Distributions, TightBindingToolkit

    using ..MeanFieldToolkit.TBMFT: TightBindingMFT
    using ..MeanFieldToolkit.BDGMFT: BdGMFT
    using ..MeanFieldToolkit.MFTIter: MFTIterator


    function SolveMFT!(mft::TightBindingMFT{T, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingBlock.params)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT{T, R, R} ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingBlock.params) + length(mft.PairingBlock.params)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        return selfcons
    end


    function SolveMFT!(mft::TightBindingMFT{T, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingBlock.params)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT{T, R, R}, fileName::String ; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50, Initial_range::Tuple{Float64, Float64} = (-0.5, 0.5)) :: SelfCons where {T, R}

        Initial     =   R.(rand(Uniform(Initial_range...), length(mft.HoppingBlock.params) + length(mft.PairingBlock.params)))
        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        return selfcons
    end


    function SolveMFT!(mft::TightBindingMFT{T, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT{T, R, R}, Initial::Vector{R}; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons ; max_iter = max_iter, tol = tol)
        return selfcons
    end

    function SolveMFT!(mft::TightBindingMFT{T, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        return selfcons
    end

    function SolveMFT!(mft::BdGMFT{T, R, R}, Initial::Vector{R}, fileName::String; Update::Function = SimpleMixing, Update_kwargs::Dict{Symbol, Any} = Dict{Symbol, Any}(:alpha => 0.5), max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) :: SelfCons where {T, R}

        selfcons    =   SelfCons(MFTIterator, Update, Initial ; F_args = (mft , ), Update_kwargs = Update_kwargs)
        
        FixedPoint!(selfcons, fileName ; max_iter = max_iter, tol = tol, checkpoint_interval = checkpoint_interval)
        return selfcons
    end


end