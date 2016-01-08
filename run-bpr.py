
import poi
from utils import Filename
from utils import setup_log
from utils import save_model 

if __name__ == "__main__":
    mdname = "bpr"
    fn = Filename("foursquare")
    setup_log(fn.log(mdname))
    train_cks = poi.load_checkins(open(fn.train))
    test_cks = poi.load_checkins(open(fn.test))

    eva = poi.Evaluation(test_cks, full=False)
    def hook(model):
        eva.assess(model)
        save_model(model, "./model/model_%s_%i.pkl" % (mdname, model.current))
        
    mf = poi.BPR(train_cks, 
                learn_rate = 0.1, 
                reg_user=0.08, 
                reg_item=0.08, 
                ) 
    mf.train(after=hook)

