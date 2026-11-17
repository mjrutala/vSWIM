import argparse
import datetime as dt

if __name__ == "__main__":
    
        
    #add command line functionality
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--start_date", type=dt.datetime.fromisoformat, default="2015-01-01", help="start date in any ISO format")
    parser.add_argument("--end_date", type=dt.datetime.fromisoformat, default="2015-01-04", help="start date in any ISO format")
    parser.add_argument("--cadence", type=int, default=3600, help="interpolation cadence in seconds")
    parser.add_argument("--params_list", type=str, nargs='+', default = ['v_mag_SW'], help="solar wind parameters to interpolate, "+\
                                                                                         "list any of:  b_x_SW, b_y_SW, b_z_SW, b_mag_SW, "+\
                                                                                         "v_x_SW, v_y_SW, v_z_SW, v_mag_SW, tp_SW,  np_SW" )

    parser.add_argument("--get_orb", action="store_true", help="include orbital information")
    #parser.add_argument("--save_model_results", action="store_true", help="save model results to csv in the data folder")
    parser.add_argument("--save_maven_data", action="store_true", help="save a copy of the original MAVEN data in the data folder")
    #parser.add_argument("--return_original", action="store_true", help="returns a copy of the original MAVEN data to user")
    parser.add_argument("--verbose", "-v", action="store_true", help="increase output verbosity")
    
    args = parser.parse_args()

    print("Now importing vSWIM-- this may be slow, depending on package versions and system architecture.")
    import vSWIM
    vSWIM.runvSWIM(startDate = args.start_date, 
                   stopDate = args.end_date, 
                   cadence = args.cadence, 
                   params = args.params_list,
                   getOrb = args.get_orb, 
                   saveModelResults = True,
                   saveMAVENData = args.save_maven_data,  
                   verbose = args.verbose)