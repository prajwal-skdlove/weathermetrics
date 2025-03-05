library(data.table)
#library("stringr")

setwd("\\\\CXA01\\Users\\jhugh\\Documents\\HT\\NCEI_data\\2024")

vc_files <- list.files()

fn_get_stationid_and_details <- function(in_stationid){
  #in_stationid <- vc_files[1000]
  dt_file <- fread(file=in_stationid)
  
  rtn_dt <- dt_file[1,c(
    "STATION",
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "NAME",
    "REPORT_TYPE",
    "CALL_SIGN"
  )]
  
  rtn_dt[,STATION := as.character(STATION)]
  rtn_dt[,COUNTRY := substr(NAME,nchar(NAME)-1,nchar(NAME))]
  rtn_dt[,US_State := ifelse(COUNTRY == "US",
                             substr(NAME,nchar(NAME)-4,nchar(NAME)-3),
                             "-")]
  return(rtn_dt)
}

lst_file_details <- lapply(vc_files, function(x) fn_get_stationid_and_details(x))

dt_file_details <- Reduce(rbind,lst_file_details)

fwrite(dt_file_details,file="..\\Station_Info.csv")


