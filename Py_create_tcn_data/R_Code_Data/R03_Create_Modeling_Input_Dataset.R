#Organize data into matrix for modeling
fn_feat_Precip <- function(in_lst,in_prefix){
    if(TRUE==FALSE){
      in_lst <- lst_mod_data[["Precip"]]
      in_prefix <- "PR"
    }
  
    dt_precip <- in_lst
    date_at_calc <- Sys.Date()
    setorder(dt_precip,-"Date_YYYYMMDD",-"HOUR")
    dt_precip[,numday_from_calc := paste("00000",as.character(difftime(
                              date_at_calc,Date_YYYYMMDD,"days")),sep="")]
    dt_precip[,numday_from_calc := substr(numday_from_calc,
                                          nchar(numday_from_calc) - 5,
                                          nchar(numday_from_calc))]
    dt_precip[,var_name :=  paste(in_prefix,"D",numday_from_calc,"H",
                                  HOUR,sep="")] 
    dt_precip[1,ref_date := date_at_calc]

    return(dt_precip[,
                     c("var_name","precip_depth_sum_mm","ref_date")])
  
}
fn_feat_RelHum <- function(in_lst,in_prefix){
  if(TRUE==FALSE){
    in_lst <- lst_mod_data[["RelHum"]]
    in_prefix <- "RH"
  }
  
  dt_relhum <- in_lst
  date_at_calc <- Sys.Date()
  setorder(dt_relhum,-"Date_YYYYMMDD",-"HOUR")
  dt_relhum[,numday_from_calc := paste("00000",as.character(difftime(
    date_at_calc,Date_YYYYMMDD,"days")),sep="")]
  dt_relhum[,numday_from_calc := substr(numday_from_calc,
                                        nchar(numday_from_calc) - 5,
                                        nchar(numday_from_calc))]
  dt_relhum[,var_name :=  paste(in_prefix,"D",numday_from_calc,"H",
                                HOUR,sep="")] 
  dt_relhum[1,ref_date := date_at_calc]
  
  return(dt_relhum[,
                   c("var_name",
                     "Max_RH_pct",
                     "Rel_Hum_pct",
                     "Mean_RH_pct",
                     "ref_date")])
  
}
fn_feat_AirTemp <- function(in_lst,in_prefix){
  if(TRUE==FALSE){
    in_lst <- lst_mod_data[["AirTemp"]]
    in_prefix <- "AT"
  }
  
  dt_airtemp <- in_lst
  date_at_calc <- Sys.Date()
  setorder(dt_airtemp,-"Date_YYYYMMDD",-"HOUR")
  dt_airtemp[,numday_from_calc := paste("00000",as.character(difftime(
    date_at_calc,Date_YYYYMMDD,"days")),sep="")]
  dt_airtemp[,numday_from_calc := substr(numday_from_calc,
                                        nchar(numday_from_calc) - 5,
                                        nchar(numday_from_calc))]
  dt_airtemp[,var_name :=  paste(in_prefix,"D",numday_from_calc,"H",
                                HOUR,sep="")] 
  dt_airtemp[1,ref_date := date_at_calc]
  
  return(dt_airtemp[,
                   c("var_name",
                     "air_temp_degrees_cels",
                     "ref_date")])
  
}





