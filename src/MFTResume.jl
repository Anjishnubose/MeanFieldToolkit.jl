module MFTResume
    export ResumeMFT!

    using FixedPointToolkit, Logging

    using ..MeanFieldToolkit.MFTIter: MFTIterator
    using ..MeanFieldToolkit.MFTRun: SolveMFT!


    function ResumeMFT!(fileName::String ; Update::Function = SimpleMixing, max_iter::Int64 = 100, tol::Float64 = 1e-6, checkpoint_interval::Int64 = 50) :: SelfCons 

        SelfConsParams  =   Dict{Symbol, Real}(:tol => tol, :max_iter => max_iter, :checkpoint_interval => checkpoint_interval)
        
        SC              =   ContinueFixedPoint!(fileName, MFTIterator, Update, SelfConsParams)
        @info "Completed!"

        return SC
    end

end