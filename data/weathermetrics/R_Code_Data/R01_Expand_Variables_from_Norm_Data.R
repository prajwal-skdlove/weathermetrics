#Getting the data for one station
#Extracting and organizing the data from various data fields
fn_get_precip <- function(in_station_id,in_dt,in_xm){
  sensor_id <- in_station_id
  dtm <- in_dt[METRIC == in_xm,c("DATE","METRIC","VALUE")]
  dtm_null <- data.table(dtm[nchar(dtm$VALUE)==0,])
  dtm <- data.table(dtm[nchar(dtm$VALUE)>0,])                     
  
  u_values <- data.table(unique(dtm$VALUE))
  colnames(u_values) <- "VALUE"
  u_values[,value_key := seq_len(.N)]
  
  #add uniq_value_id back to original data table
  dtm <- merge(dtm,u_values,by="VALUE",all.x=TRUE)[,c("DATE","value_key")]
  
  u_values[,period_hrs := as.character(substr(VALUE,1,2))]
  u_values[,depth_mm := as.character(substr(VALUE,4,7))]
  u_values[,depth_scale := 10]
  u_values[,condition_code := as.character(substr(VALUE,9,9))]
  u_values[,quality_code := as.character(substr(VALUE,11,11))]
  u_values <- cbind(in_xm,u_values)[,c(-2)]
  
  rtn_values <- list(sensor_id,dtm,dtm_null,u_values)
  names(rtn_values) <- c("STATION","DATE_VALUE_ID","No_VALUE_RECORDS","VALUES")
  return(rtn_values)
}
fn_get_rel_humidity <- function(in_station_id,in_dt,in_xm){
  sensor_id <- in_station_id
  dtm <- in_dt[METRIC == in_xm,c("DATE","METRIC","VALUE")]
  dtm_null <- data.table(dtm[nchar(dtm$VALUE)==0,])
  dtm <- data.table(dtm[nchar(dtm$VALUE)>0,])                     
  
  u_values <- data.table(unique(dtm$VALUE))
  colnames(u_values) <- "VALUE"
  u_values[,value_key := seq_len(.N)]

  #add uniq_value_id back to original data table
  dtm <- merge(dtm,u_values,by="VALUE",all.x=TRUE)[,c("DATE","value_key")]
  
  u_values[,period_hrs := as.character(substr(VALUE,1,3))]
  u_values[,period_scale := 1]
  u_values[,rh_code := as.character(substr(VALUE,5,5))]
  u_values[,rh_pct := as.character(substr(VALUE,7,9))]
  u_values[,rh_scale := 1]
  u_values[,rh_derived_code := as.character(substr(VALUE,11,11))]
  u_values[,quality_code := as.character(substr(VALUE,13,13))]
  u_values <- cbind(in_xm,u_values)[,c(-2)]
  
  rtn_values <- list(sensor_id,dtm,dtm_null,u_values)
  names(rtn_values) <- c("STATION","DATE_VALUE_ID","No_VALUE_RECORDS","VALUES")
  return(rtn_values)
}
fn_get_temp <- function(in_station_id,in_dt,in_xm){
  sensor_id <- in_station_id
  dtm <- in_dt[METRIC == in_xm,c("DATE","METRIC","VALUE")]
  dtm_null <- data.table(dtm[nchar(dtm$VALUE)==0,])
  dtm <- data.table(dtm[nchar(dtm$VALUE)>0,])                     
  
  u_values <- data.table(unique(dtm$VALUE))
  colnames(u_values) <- "VALUE"
  u_values[,value_key := seq_len(.N)]
  
  #add uniq_value_id back to original data table
  dtm <- merge(dtm,u_values,by="VALUE",all.x=TRUE)[,c("DATE","value_key")]
  
  u_values[,degrees_cels := as.character(substr(VALUE,1,5))]
  u_values[,degrees_scale := 10]
  u_values[,quality_code := as.character(substr(VALUE,7,7))]
  u_values <- cbind(in_xm,u_values)[,c(-2)]
  
  rtn_values <- list(sensor_id,dtm,dtm_null,u_values)
  names(rtn_values) <- c("STATION","DATE_VALUE_ID","No_VALUE_RECORDS","VALUES")
  return(rtn_values)
}
fn_get_air_pressure <- function(in_station_id,in_dt,in_xm){
  sensor_id <- in_station_id
  dtm <- in_dt[METRIC == in_xm,c("DATE","METRIC","VALUE")]
  dtm_null <- data.table(dtm[nchar(dtm$VALUE)==0,])
  setorder(dtm_null,"DATE")
  dtm <- data.table(dtm[nchar(dtm$VALUE)>0,])                     
  
  u_values <- data.table(unique(dtm$VALUE))
  colnames(u_values) <- "VALUE"
  u_values[,value_key := seq_len(.N)]
  
  #add uniq_value_id back to original data table
  dtm <- merge(dtm,u_values,by="VALUE",all.x=TRUE)[,c("DATE","value_key")]
  setorder(dtm,"DATE")
  
  u_values[,sea_level_pressure_hectopascals := as.character(substr(VALUE,1,5))]
  u_values[,sea_level_pressure_scale := 10]
  u_values[,quality_code := as.character(substr(VALUE,7,7))]
  u_values <- cbind(in_xm,u_values)[,c(-2)]
  
  rtn_values <- list(sensor_id,dtm,dtm_null,u_values)
  names(rtn_values) <- c("STATION","DATE_VALUE_ID","No_VALUE_RECORDS","VALUES")
  return(rtn_values)
}
fn_get_wind <- function(in_station_id,in_dt,in_xm){
  sensor_id <- in_station_id
  dtm <- in_dt[METRIC == in_xm,c("DATE","METRIC","VALUE")]
  dtm_null <- data.table(dtm[nchar(dtm$VALUE)==0,])
  setorder(dtm_null,"DATE")
  dtm <- data.table(dtm[nchar(dtm$VALUE)>0,])                     
  
  u_values <- data.table(unique(dtm$VALUE))
  colnames(u_values) <- "VALUE"
  u_values[,value_key := seq_len(.N)]
  
  #add uniq_value_id back to original data table
  dtm <- merge(dtm,u_values,by="VALUE",all.x=TRUE)[,c("DATE","value_key")]
  setorder(dtm,"DATE")
  
  u_values[,direction_angle_degrees := as.character(substr(VALUE,1,3))]
  u_values[,direction_angle_quality_code := as.character(substr(VALUE,5,5))]
  u_values[,type_code := as.character(substr(VALUE,7,7))]
  u_values[,speed_meter_per_sec := as.character(substr(VALUE,9,12))]
  u_values[,speed_scale := 10]
  u_values[,speed_quality_code := as.character(substr(VALUE,14,14))]
  u_values <- cbind(in_xm,u_values)[,c(-2)]
  
  rtn_values <- list(dtm,dtm_null,u_values)
  
  rtn_values <- list(sensor_id,dtm,dtm_null,u_values)
  names(rtn_values) <- c("STATION","DATE_VALUE_ID","No_VALUE_RECORDS","VALUES")
  return(rtn_values)
}
fn_get_ceiling <- function(in_station_id,in_dt,in_xm){
  sensor_id <- in_station_id
  dtm <- in_dt[METRIC == in_xm,c("DATE","METRIC","VALUE")]
  dtm_null <- data.table(dtm[nchar(dtm$VALUE)==0,])
  setorder(dtm_null,"DATE")
  dtm <- data.table(dtm[nchar(dtm$VALUE)>0,])                     
  
  u_values <- data.table(unique(dtm$VALUE))
  colnames(u_values) <- "VALUE"
  u_values[,value_key := seq_len(.N)]
  
  #add uniq_value_id back to original data table
  dtm <- merge(dtm,u_values,by="VALUE",all.x=TRUE)[,c("DATE","value_key")]
  setorder(dtm,"DATE")

  u_values[,ceiling_height_meters := as.character(substr(VALUE,1,5))]
  u_values[,ceiling_height_scale := 1]
  u_values[,ceiling_height_quality_code := as.character(substr(VALUE,7,7))]
  u_values[,ceiling_height_determine_code := as.character(substr(VALUE,9,9))]
  u_values[,CAVOK_code := as.character(substr(VALUE,11,11))]
  u_values <- cbind(in_xm,u_values)[,c(-2)]
  
  rtn_values <- list(dtm,dtm_null,u_values)
  
  rtn_values <- list(sensor_id,dtm,dtm_null,u_values)
  names(rtn_values) <- c("STATION","DATE_VALUE_ID","No_VALUE_RECORDS","VALUES")
  return(rtn_values)
}
fn_aggregate_metrics <- function(in_setwd,in_xstation,in_dt){
  
  setwd(in_setwd)
  
  lst_precip <- lapply(c("AA1","AA2","AA3","AA4"), function(xm)
    fn_get_precip(in_xstation,in_dt,xm))
  
  lst_rel_humidity <- lapply(c("RH1","RH2","RH3"), function(xm)
    fn_get_rel_humidity(in_xstation,in_dt,xm))
  
  lst_temp <- lapply(c("TMP"), function(xm)
    fn_get_temp(in_xstation,in_dt,xm))
  
  lst_air_pressure <- lapply(c("SLP"), function(xm)
    fn_get_air_pressure(in_xstation,in_dt,xm))
  
  lst_wind <- lapply(c("WND"), function(xm)
    fn_get_wind(in_xstation,in_dt,xm))
  
  lst_ceiling <- lapply(c("CIG"), function(xm)
    fn_get_ceiling(in_xstation,in_dt,xm))
  
  assign("lst_metrics",
         list(lst_precip[[1]],
              lst_precip[[2]],
              lst_precip[[3]],
              lst_rel_humidity[[1]],
              lst_rel_humidity[[2]],
              lst_rel_humidity[[3]],
              lst_temp,
              lst_air_pressure,
              lst_wind,
              lst_ceiling))
  
  names(lst_metrics) <- c("Precip1",
                          "Precip2",
                          "Precip3",
                          "RelHum1",
                          "RelHum2",
                          "RelHum3",
                          "Temp",
                          "AirPressure",
                          "Wind",
                          "Ceiling")
  
  return(lst_metrics)
}

