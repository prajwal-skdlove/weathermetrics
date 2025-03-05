#Getting the data for one station
#Cleaning and preparing data from various data fields
#For modeling
#dt_AA1 <- lst_metrics[["Precip1"]]
#dt_AA2 <- lst_metrics[["Precip2"]]
#dt_AA3 <- lst_metrics[["Precip3"]]

fn_moddat_precip <- function(dt_AA1,dt_AA2,dt_AA3){
  quality_codes <- c(0,1,4,5,9) #Keep these codes
  
  lst_in_dt <- list(dt_AA1,dt_AA2,dt_AA3)

  #Uniquely get all DATE values
  dt_all_DATE <- Reduce(rbind,lapply(seq_along(lst_in_dt),
                        function(x) {
                          dt <- lst_in_dt[[x]]$DATE_VALUE_ID
                          dt[,"AA_var" := x]
                          dt
                        }))
  
  #Uniquely get all metric values
  dt_all_VALUES <- Reduce(rbind,lapply(seq_along(lst_in_dt),
                                     function(x) {
                                       dt <- lst_in_dt[[x]]$VALUES
                                       dt[,"AA_var" := x]
                                       dt
                                     }))
  
  
  #Merge with values and select those with 1 hour periods
  dt_all_DATE_Values <- merge(dt_all_DATE,
                              dt_all_VALUES,
                              by=c("AA_var","value_key"),
                              all.x=TRUE)
  

  
  
  
  #Select those with period_hrs equal to 01
  dt_all_DATE_Values_01hr <- data.table(dt_all_DATE_Values[period_hrs == "01",])
  dt_all_DATE_Values_01hr[,Date_YYYYMMDD := substr(DATE,1,10)]  
  dt_all_DATE_Values_01hr[,HOUR := ifelse(hour(DATE)<10,paste("0",
                              hour(DATE),sep=""),
                              as.character(hour(DATE)))]
  dt_all_DATE_Values_01hr[,dup_DayHour := .N,
                          by=c("Date_YYYYMMDD","HOUR","value_key")]
  table(dt_all_DATE_Values_01hr$dup_DayHour)
  
  #Take out records with duplicate value keys inj same hour asssuming that the 
  #info was accidentally put in more than once
  dt_uniq <- data.table(dt_all_DATE_Values_01hr[dup_DayHour == 1,])
  dt_dupes <- data.table(unique(dt_all_DATE_Values_01hr[dup_DayHour > 1,]))
  dt_dupes[,first_record_date := min(DATE),
           by=c("Date_YYYYMMDD","HOUR","value_key")]
  dt_dupes <- dt_dupes[first_record_date == DATE,]
  
  dt_all_DATE_Values_01hr <- rbind(dt_uniq[,1:11],dt_dupes[,1:11])
  setorder(dt_all_DATE_Values_01hr,"Date_YYYYMMDD","HOUR")
  
  
  #Create reference dt that contains all dates and all hours
  if(dim(dt_all_DATE_Values_01hr)[1] >1){
    xst <- as.Date(min(dt_all_DATE_Values_01hr$Date_YYYYMMDD))
    xnd <- as.Date(max(dt_all_DATE_Values_01hr$Date_YYYYMMDD))
    
    vc_all_dates <- seq.Date(xst,xnd,by = "day")
    
    vc_all_hrs <- c(paste("0",seq(0,9),sep=""),seq(10,23,1))
    
    dt_date_hour_ref <- data.table(merge(as.character(vc_all_dates), vc_all_hrs))
    setorder(dt_date_hour_ref,"x","y")
    colnames(dt_date_hour_ref) <- c("Date_YYYYMMDD","HOUR")
    
    #Merge the observed dt with the reference dt
    dt_ref_values <- merge(dt_date_hour_ref,
                          dt_all_DATE_Values_01hr,
                          by=c("Date_YYYYMMDD","HOUR"),
                          all.x=TRUE)
    #Apply codes to eliminate bad measurements
    dt_ref_values <- dt_ref_values[quality_code %in% quality_codes,]
    
    
    #Keep good records only records
    dt_ref_values <- data.table(dt_ref_values[depth_mm < 9999 &
                                                depth_mm >= 0,])
    
    #Calculate the rainfall amount
    dt_ref_values[,precip_depth_mm := as.numeric(depth_mm)/as.numeric(depth_scale)]
    
    #Sum the precip depth to get full amount for hour
    dt_ref_values[, precip_depth_sum_mm := sum(precip_depth_mm),
                  by=c("Date_YYYYMMDD","HOUR")]
    dt_ref_values[, check_gt1_records := .N,
                  by=c("Date_YYYYMMDD","HOUR")]
    
    tbl_num_records <- tabulate(dt_ref_values$check_gt1_records)
    
    dt_return0 <- data.table(unique(dt_ref_values[,c(
                                  "Date_YYYYMMDD",
                                  "HOUR",
                                  "period_hrs",
                                  "precip_depth_sum_mm"
                                  )]))
    
    dt_return <- merge(dt_date_hour_ref,
          dt_return0,
          by=c("Date_YYYYMMDD","HOUR"),
          all.x=TRUE)
    
    return(dt_return)
  } else { 
    return(dt_all_DATE_Values_01hr)
    }
}
fn_moddat_rel_humidity <- function(dt_RH1,dt_RH2,dt_RH3){
  quality_codes <- c(0,1,4,5) #Keep these codes
  
  #RH1 is maximum relative humidity X
  #RH2 is relative humidity         N
  #RH3 is mean relative humidity    M
  lst_in_dt <- list(dt_RH1,dt_RH2,dt_RH3)

  #Uniquely get all DATE values
  lst_metric <- c("Max_RH","Rel_Hum","Mean_RH")
  dt_all_DATE <- Reduce(rbind,lapply(seq_along(lst_in_dt),
                                     function(x) {
                                       dt <- lst_in_dt[[x]]$DATE_VALUE_ID
                                       dt[,"RH_var" := lst_metric[x]]
                                       dt
                                     }))

    
  #Uniquely get all metric values
  dt_all_VALUES <- Reduce(rbind,lapply(seq_along(lst_in_dt),
                            function(x) {
                              dt <- lst_in_dt[[x]]$VALUES
                              dt[,"RH_var" := lst_metric[x]]
                              dt[,RH_pct := 
                                  as.numeric(rh_pct)/as.numeric(rh_scale)]
                              dt[,c(
                                "RH_var",
                                "value_key",
                                "RH_pct",
                                "quality_code"
                              )]
                            }))
  
    
  #Merge with values and select those with 1 hour periods
  dt_all_DATE_Values <- merge(dt_all_DATE,
                              dt_all_VALUES,
                              by=c("RH_var","value_key"),
                              all.x=TRUE)
  
  
  #Apply codes to eliminate bad measurements
  dt_all_DATE_Values[,RH_pct := ifelse(quality_code %in% quality_codes,RH_pct,NA)]
  dt_all_DATE_Values$quality_code <- NULL
  
  #RH is only measured once a day
  dt_all_DATE_Values_24hr <- data.table(DATE=unique(dt_all_DATE_Values$DATE))
  dt_all_DATE_Values_24hr[,Date_YYYYMMDD := substr(DATE,1,10)]  
  dt_all_DATE_Values_24hr[,HOUR := ifelse(hour(DATE)<10,paste("0",
                                                              hour(DATE),sep=""),
                                          as.character(hour(DATE)))]
  setorder(dt_all_DATE_Values_24hr,"DATE")
  
  #Create the metrics for each value
  lst_metrics_values <- lapply(seq_along(lst_metric), function(x) {
      dt <- dt_all_DATE_Values[RH_var == lst_metric[x],]
      dt <- dt[,c("DATE","RH_pct")]
      colnames(dt) <- c("DATE",paste(lst_metric[x],"_pct",sep=""))
      dt
    })
  

  dt_ref_values <- Reduce(function(d1,d2)
                          merge(d1,d2,by="DATE",all.x=TRUE),
                          c(list(dt_all_DATE_Values_24hr),lst_metrics_values))
  dt_ref_values[,period_hrs := "24"]
  
  dt_return <- dt_ref_values[,c(2:3,7,4:6)]
    
  return(dt_return)
}
fn_moddat_temp <- function(dt_Temp){
  # dt_Temp <- lst_metrics[["Temp"]]
  quality_codes <- c(0,1,4,5,9) #Keep these codes
  
  lst_in_dt <- dt_Temp
  
  #Uniquely get all DATE values
  dt_all_DATE <- data.table(DATE=unique(c(lst_in_dt[[1]]$DATE_VALUE_ID$DATE,
                            lst_in_dt[[1]]$NO_VALUE_RECORDS$DATE)))
  dt_all_DATE <- merge(dt_all_DATE,
                       lst_in_dt[[1]]$DATE_VALUE_ID,
                       by="DATE",
                       all.x=TRUE)  
  
  #Identify DATEs with multiple measurements
  dt_all_DATE[,multi_metrics := .N,by="DATE"]

  #Create keys for summarizing information
  dt_all_DATE[,Date_YYYYMMDD := substr(DATE,1,10)]  
  dt_all_DATE[,HOUR := ifelse(hour(DATE)<10,paste("0",
                            hour(DATE),sep=""),
                            as.character(hour(DATE)))]
  setorder(dt_all_DATE,"Date_YYYYMMDD","HOUR")

  #Uniquely get all metric values that are valid
  dt_all_VALUES <- data.table(lst_in_dt[[1]]$VALUES)
  dt_all_VALUES <- data.table(dt_all_VALUES[quality_code %in% quality_codes,])
  #Eliminate bad data
  dt_all_VALUES <- dt_all_VALUES[ degrees_cels != "+9999",]
  dt_all_VALUES[,air_temp_degrees_cels := 
                  as.numeric(degrees_cels)/as.numeric(degrees_scale)]
  
  #Merge with values and select those with 1 hour periods
  dt_all_DATE_Values <- merge(dt_all_DATE,
                              dt_all_VALUES,
                              by=c("value_key"),
                              all.x=TRUE)

  #Clean up dupes
  dt_all_DATE_Values_nodupes <- as.data.table(unique(dt_all_DATE_Values[,
                                                    c("Date_YYYYMMDD",
                                                      "HOUR",
                                                      "air_temp_degrees_cels")]))
  
  dt_all_DATE_Values_nodupes[,air_temp_mean := mean(air_temp_degrees_cels,na.rm=TRUE),
               by=c("Date_YYYYMMDD","HOUR")]

  dt_ret_values <- data.table(unique(
                      dt_all_DATE_Values_nodupes[,c(1,2,4)]))
  
  colnames(dt_ret_values) <- colnames(dt_all_DATE_Values_nodupes)[1:3]
  
  dt_ret_values[,multi_metrics := .N,by=c("Date_YYYYMMDD",
                                           "HOUR")]
  dt_ret_values[,period_hrs := "01"]
  
  setorder(dt_ret_values,"Date_YYYYMMDD","HOUR")
  
  return(dt_ret_values[,c(1,2,5,3)])
}

