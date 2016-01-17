# -*- coding: utf-8 -*-

import poi
from utils import *

if __name__ == "__main__":
    setup_log()
    fn = Filename("gowalla")
    locations = poi.load_locations(open(fn.locations)) 
    cks = poi.load_checkins(open(fn.train))

    pl = poi.PowerLaw(cks, locations)
    pl.count()
    pl.guass(max_x=10.0)
    pl.plot("gw_pic.png")
    # save pl model to file for reuse.
    save_model(pl, "model/gw_powerlaw.pkl")
