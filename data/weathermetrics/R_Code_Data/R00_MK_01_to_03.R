#Run data for a station
library(data.table)

#Location of r code files
ip <- "C:\\Users\\jhugh\\Documents\\GitHub\\Py_Weather_S4\\weathermetrics\\data\\weathermetrics\\R_Code_Data"
setwd(ip)

source("R01_Expand_Variables_from_Norm_Data.R")
source("R02_Create_Modeling_Data.R")
source("R03_Create_Modeling_Input_Dataset.R")

fn_create_model_data <- function(in_wd,out_wd,in_xstation){
# in_xstation <- vc_stations[14]
setwd(in_wd)

########################
#1. Read normalized data
#dt_metrics_year_norm <- fread(file=paste(in_xstation,"metrics_norm.cvs",sep="_"))
##Create metrics file
#lst_metrics <- fn_aggregate_metrics(wd,in_xstation,dt_metrics_year_norm)
#fn <- paste(in_xstation,"_lst_metrics",".RDS",sep="")
#save(lst_metrics,file=fn)

#Get lst_metrics file
fn <- paste(in_wd,"\\",in_xstation,"_lst_metrics",".RDS",sep="")
load(file=fn)

###################################
#2. Create all the modeling metrics
precip_model_data <- fn_moddat_precip(lst_metrics[["Precip1"]],
                                      lst_metrics[["Precip2"]],
                                      lst_metrics[["Precip3"]])

rel_humidity_model_data <- fn_moddat_rel_humidity(lst_metrics[["RelHum1"]],
                                                  lst_metrics[["RelHum2"]],
                                                  lst_metrics[["RelHum3"]])

air_temp_model_data <- fn_moddat_temp(lst_metrics[["Temp"]])


lst_mod_data <- list(precip_model_data,
                     rel_humidity_model_data,
                     air_temp_model_data)
names(lst_mod_data) <- c("Precip","RelHum","AirTemp")

fn <- paste(in_xstation,"_lst_mod_dat",".rds",sep="")
save(lst_mod_data,file=fn)

##################################
#3. Create csv data sets for outputing
fn <- paste(in_xstation,"_lst_mod_dat",".RDS",sep="")
load(file=fn)

#out_name <- paste("model_data_precip",sep="")
#dt_precip_data_values <- fn_feat_Precip(lst_mod_data[["Precip"]],"PR")

#Write out precip data column
dt00 <- lst_mod_data[["Precip"]]
if(nrow(dt00) > 1){
fwrite(fn_feat_Precip(dt00,"PR"),
       file=paste(out_wd,"\\",paste(in_xstation,"_model_data_precip",sep=""),".csv",
                  sep=""))}

dt00 <- lst_mod_data[["RelHum"]]
if(nrow(dt00) > 1){
fwrite(fn_feat_RelHum(dt00,"RH"),
       file=paste(out_wd,"\\",paste(in_xstation,"_model_data_relhum",sep=""),".csv",
                  sep=""))}

dt00 <- lst_mod_data[["AirTemp"]]
if(nrow(dt00) > 1){
fwrite(fn_feat_AirTemp(dt00,"AT"),
       file=paste(out_wd,"\\",paste(in_xstation,"_model_data_airtemp",sep=""),".csv",
                  sep=""))
  }

return(1)
}


#Location of r data files for each station
iwd <- "C:\\Users\\jhugh\\Documents\\HT\\NCEI_data"

#Location where output csv files go in advance of creating Py data matrices
owd <- "C:\\Users\\jhugh\\Documents\\HT\\NCEI_data\\metrics_csv"
setwd(iwd)
dt_stations <- fread(file="Station_Info.csv")

#in_state <- c("NY","MD", "VA","GA","KY","OH","TN")
#in_state <- c("NH","VT","ME","CT","RI","MA","NJ","PA")


vc_stations <- dt_stations[US_State %in% c(in_state),STATION]

lst_complete <- lapply(vc_stations, function(in_xstation) 
                    fn_create_model_data(iwd,owd,in_xstation))

