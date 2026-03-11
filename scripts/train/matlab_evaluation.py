import matlab
import matlab.engine
import numpy as np 
import torch



class matlab_eval():
    def __init__(self, sofa_path):
        
        self.eng = matlab.engine.start_matlab()
        amt_path = 'C:\\Users\\as06521\\OneDrive - University of Surrey\\Documents\\Research\\Code\\Matlab\\model_comparison_image_shift-main\\amtoolbox-1.6.0'
        self.eng.addpath(amt_path)
        self.eng.eval("amt_start()", nargout = 0)

        self.sofa_path = sofa_path

        ### DIETZ VARIABLES

        self.dietz_lookup = None


    def dietz_eval_init(self):
        sofaile = self.eng.SOFAload(self.sofa_path, nargout = 1)
        self.fs = sofaile.Data.SamplingRate
        self.dietz_lookup = self.eng.itd2angle(sofaile, self.fs)

    def dietz_eval(self, insig):
        
        if self.dietz_lookup == None:
            self.dietz_eval_init()

        insig_np = insig.detach().cpu().numpy().tolist()
        out_dietz2011, fc_dietz2011, ild_dietz2011, env_dietz2011 = self.eng.dietz2011(matlab.double(insig_np))

