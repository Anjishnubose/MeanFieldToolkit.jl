module Build

    export BuildFromInteractions!

    using LinearAlgebra, TightBindingToolkit
    
    using ..MeanFieldToolkit.MFTBonds: GetMFTBonds, GetBondDictionary
    using ..MeanFieldToolkit.TBMFT:TightBindingMFT
    using ..MeanFieldToolkit.BDGMFT:BdGMFT


    ##### ///TODO: Add Free Hopping energies also
    function BuildFromInteractions!(tbMFT::TightBindingMFT{T} ; refresh::Bool = true) where {T}

        if refresh
            labels 	=	getproperty.(tbMFT.TightBindingModel.uc.bonds, :label)
            deleteat!(tbMFT.TightBindingModel.uc.bonds, findall(startswith.(labels, "MFT =>")))
        end

        for (i, InteractionBlock) in enumerate(tbMFT.InteractionBlock)

            labels      =   Dict{String, String}()
            for key in ["ii", "jj", "ij"]
                labels[key]     =   "Interaction $(i) -> $(tbMFT.ChannelLabels[key])"
            end

            for BondKey in keys(InteractionBlock.lookup)
                
                Expectations        =   GetBondDictionary(tbMFT.HoppingBlock.lookup, BondKey, tbMFT.HoppingBlock.uc.localDim)
                Decomposed          =   tbMFT.MFTDecomposition[i](InteractionBlock.lookup[BondKey] , Expectations)

                MFTBonds            =   GetMFTBonds(Decomposed ; BondKey = BondKey, uc = tbMFT.HoppingBlock.uc, scaling = tbMFT.MFTScaling, labels = labels)
                append!(tbMFT.TightBindingModel.uc.bonds, MFTBonds)

            end
        end

    end

    ##### ///TODO: Add Free Hopping and pairing energies also
    function BuildFromInteractions!(bdgMFT::BdGMFT{T} ; refresh::Bool = true) where {T}

        if refresh
            labels 	=	getproperty.(bdgMFT.bdgModel.uc_hop.bonds, :label)
            deleteat!(bdgMFT.bdgModel.uc_hop.bonds, findall(startswith.(labels, "MFT =>")))

            labels 	=	getproperty.(bdgMFT.bdgModel.uc_pair.bonds, :label)
            deleteat!(bdgMFT.bdgModel.uc_pair.bonds, findall(startswith.(labels, "MFT =>")))
        end

        for (i, InteractionBlock) in enumerate(bdgMFT.InteractionBlock)

            HoppingLabels      =   Dict{String, String}()
            PairingLabels      =   Dict{String, String}()
            for key in ["ii", "jj", "ij"]
                HoppingLabels[key]     =   "Interaction $(i) -> $(bdgMFT.HoppingLabels[key])"
                PairingLabels[key]     =   "Interaction $(i) -> $(bdgMFT.PairingLabels[key])"
            end

            for BondKey in keys(InteractionBlock.lookup)
            
                HoppingExpectations =   GetBondDictionary(bdgMFT.HoppingBlock.lookup, BondKey, bdgMFT.HoppingBlock.uc.localDim)
                PairingExpectations =   GetBondDictionary(bdgMFT.PairingBlock.lookup, BondKey, bdgMFT.PairingBlock.uc.localDim)

                DecomposedHopping   =   bdgMFT.HoppingDecomposition[i](InteractionBlock.lookup[BondKey] , HoppingExpectations)
                DecomposedPairing   =   bdgMFT.PairingDecomposition[i](InteractionBlock.lookup[BondKey] , PairingExpectations)

                HoppingBonds        =   GetMFTBonds(DecomposedHopping ; BondKey = BondKey, uc = bdgMFT.HoppingBlock.uc, scaling = bdgMFT.HoppingScaling, labels = HoppingLabels)
                append!(bdgMFT.bdgModel.uc_hop.bonds, HoppingBonds)

                PairingBonds        =   GetMFTBonds(DecomposedPairing ; BondKey = BondKey, uc = bdgMFT.PairingBlock.uc, scaling = bdgMFT.PairingScaling, labels = PairingLabels)
                append!(bdgMFT.bdgModel.uc_pair.bonds, PairingBonds)

            end

        end

    end



end