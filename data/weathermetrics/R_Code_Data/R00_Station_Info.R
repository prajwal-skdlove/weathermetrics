fn_get_stationid_and_details <- function(in_stationid){
  
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

fn_get_nearby_stations <- function(in_dt,in_stid,in_deg_distant){
  # in_dt the input data set from the list of all stations
  # in_stid the station id of the target station
  # how close in degrees the other stations need to be    
  trg_station_id <- in_stid
  trg_lats <- in_dt[STATION == in_stid,LATITUDE]
  trg_long <- in_dt[STATION == in_stid,LONGITUDE]

  in_dt[,lats_dist := abs(LATITUDE - trg_lats)]
  in_dt[,long_dist := abs(LONGITUDE - trg_long)]

  in_dt[,lats_close := lats_dist < in_deg_distant]
  in_dt[,long_close := long_dist < in_deg_distant]

  in_dt[,vect_close := sqrt(lats_dist*lats_dist 
                      + long_dist*long_dist)]

  rtn_vals <- in_dt[vect_close < in_deg_distant ,]
  rtn_vals[,target_station := trg_station_id]
  rtn_vals[,vect_close_bkt := cut(vect_close,
      breaks=seq(0, in_deg_distant, by = 0.5))]

  rtn_vals[,bearing := fn_bearing(trg_long,trg_lats,
                                LONGITUDE,LATITUDE
                          )]

  rtn_vals[,bearing_bkt := cut(bearing,
      breaks=seq(0, 360, by = 30))]

  keep_var <- c("target_station",
                "STATION",
                "vect_close_bkt",
                "bearing_bkt"
  )

#return(rtn_vals[,c("STATION","target_station","vect_close_bkt")])

#return(rtn_vals[!is.na(vect_close_bkt),..keep_var])
return(rtn_vals[,..keep_var])
}

fn_bearing <- function(lon1, lat1, lon2, lat2) {
  lon1 <- lon1 * pi / 180
  lat1 <- lat1 * pi / 180
  lon2 <- lon2 * pi / 180
  lat2 <- lat2 * pi / 180
  
  y <- sin(lon2 - lon1) * cos(lat2)
  x <- cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
  
  bearing <- atan2(y, x)
  bearing_degrees <- (bearing * 180 / pi + 360) %% 360
  
  return(bearing_degrees)
}

fn_has_metrics <- function(in_path,in_stations,in_metrics){
  # in_path place where all the metric csv files are kept
  # in_stations the target station and the nearby stations
  # in_metrics is a list of the metrics that need to be available for all stations
  dt_stations <- data.table(fnames = list.files(path = in_path))

  dt_stations[, st_id := substr(fnames,1,11)]

  trgt_station <- in_stations$target_station[1]

  dt_desired_stations <- merge(in_stations,dt_stations,
                  by.x="STATION",
                  by.y="st_id",
                  all.x=TRUE)

  dt_desired_stations <- dt_desired_stations[!is.na(fnames),]
  dt_desired_stations[,has_metric := 0]

  lst_metrics <- lapply(in_metrics, function(x) {
    index_metric <- grep(x,dt_desired_stations$fnames)
    })

  for(i in 1:length(lst_metrics)){
      dt_desired_stations[lst_metrics[[i]],has_metric := 1]
  }

num_required_metrics <- length(lst_metrics)

dt_desired_stations[, cnt_metrics := sum(has_metric), by="STATION"]

dt_rtn <- data.table(dt_desired_stations[cnt_metrics == num_required_metrics &
                      has_metric == 1,])

dt_rtn[,st_id := STATION]
dt_rtn[,target_station := trgt_station]
dt_rtn[,path := in_path]  

rtn_values <- dt_rtn[,c("target_station","st_id","fnames",
"path","vect_close_bkt","bearing_bkt")]

return(rtn_values)
}

fn_rank_stations <- function(in_dt_nearby_stations){

  dt_return <- in_dt_nearby_stations
  dt_return[,vect_close_bkt2 := 1+as.numeric(ifelse(is.na(vect_close_bkt),"0",vect_close_bkt))]
  setorder(dt_return,"vect_close_bkt2","fnames")

#  num_measures <- tabulate(as.numeric(dt_return$vect_close_bkt == 0))
#  num_sets <- length(dt_return$vect_close_bkt)/num_measures
#  dt_return$station_rank <- rep(1:19,3)[order(rep(1:num_sets,num_measures))]

  return(dt_return)
}

fn_list_stations_to_include <- function(in_ipd01,in_ipd02,
                                        in_ipfn,in_st_id,
                                        in_v_dist_degrees,in_metrics){

  setwd(in_ipd01)

  dt_file_details <- fread(file= in_ipfn)
  
  lst_nearby_stations <- fn_get_nearby_stations(
          dt_file_details,
          in_st_id,
          in_v_dist_degrees)


  dt_nearby_stations <- fn_has_metrics(in_ipd02,lst_nearby_stations,in_metrics)

  dt_ranked_nearby_stations <- fn_rank_stations(dt_nearby_stations)

  return(dt_ranked_nearby_stations)
}



#Identify nearby stations and check if they have metric files
library(data.table)

ipd01 <- "\\\\CXA01\\Users\\jhugh\\Documents\\HT\\NCEI_data\\2024"
ipd02 <- "\\\\CXA01\\Users\\jhugh\\Documents\\HT\\NCEI_data\\metrics_csv"
ipfn <- "..\\Station_Info.csv"
#in_st_id <- "72502014734"

#estimate of miles
#To get latitude in radians multiply degrees by pi/180
#distance = 69.17 miles * cos(latitude)
in_v_dist_degrees <- 3
in_metrics <- list("precip", "relhum", "airtemp")


lst_target_stations <- c(
  "74486094789",
  "72518014735",
  "72508014740",
  "72526014860",
  "72406093721",
  "72401013740",
  "72219013874",
  "72423093821",
  "72428014821",
  "72327013897"
)

lst_output <- lapply(lst_target_stations, function(x) fn_list_stations_to_include(
          ipd01,
          ipd02,
          ipfn,
          x,
          in_v_dist_degrees,
          in_metrics))

dt_output <- Reduce(rbind,lst_output)

out_file <- paste(ipd02,"\\","Target_station_analysis_set",".csv",sep="") 
fwrite(dt_output,file=out_file)






