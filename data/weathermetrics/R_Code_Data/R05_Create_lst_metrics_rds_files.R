
#Pull out data one sensor at a time
fn_get_sensor_files <- function(in_year,fn){
  list_of_files <- untar(paste(in_year,".tar.gz",sep=""),list=TRUE)
  Has_file <- fn %in% list_of_files

  if(Has_file){
    untar(paste(in_year,".tar.gz",sep=""),files=fn)
    dt_fn <- fread(fn)
    file.remove(fn)
    rtn_values <- dt_fn
  } else {
    rtn_values <- NULL
  }
  return(rtn_values)
}

fn_create_norm_metrics_files <- function(in_wd,in_yrs,in_station){
if(TRUE==FALSE){
  in_yrs <- lst_years
  in_station <- vc_stations[7]
}
#Get worksheets over time for a station
print(Sys.time())
lst_dtfiles <- lapply(in_station, function(xstation){
                  sensor_file <- paste(xstation,".csv",sep="")
                  lst_dt_fn <- lapply(in_yrs, function(x) 
                  fn_get_sensor_files(x,sensor_file))
                  lst_dt_fn
  })


#Normalize data for each dt
#Drop repetitive fields
drop_var <- c("SOURCE",
              "LATITUDE",
              "LONGITUDE",
              "ELEVATION",
              "NAME",
              "REPORT_TYPE",
              "CALL_SIGN",
              "QUALITY_CONTROL")

#Identify the years missing data
lst_has_metrics <- Reduce(rbind,lapply(seq_along(in_yrs), 
                        function(xy){ !is.null(lst_dtfiles[[1]][[xy]])
  }))
print(Sys.time())

#name the list elements
names(lst_dtfiles[[1]]) <- in_yrs
#Identify the elements with data and only keep those
keep_lst_elem <- names(lst_dtfiles[[1]])[lst_has_metrics]

lst_dt_fn <- lst_dtfiles[[1]][keep_lst_elem]

dt_metrics_year_norm <- Reduce(rbind,lapply(seq_along(lst_dt_fn), 
    function(xy) {
      if(is.null(lst_dt_fn[[xy]])){
      
      }else{
        dtf01 <- data.table(lst_dt_fn[[xy]])
        keep_var <- setdiff(colnames(dtf01),drop_var)
        dtf02 <- dtf01[,..keep_var]
    
        dt_metrics_norm <- Reduce(rbind,lapply(3:length(colnames(dtf02)),
                           function(x) {
                             dtf03 <- dtf02[,c(1,2,..x)]
                             dtf03[,Metric := colnames(dtf03)[3]]
                             new_colnames <- 
                               c(colnames(dtf03)[1:2],"VALUE","METRIC")
                             colnames(dtf03) <- new_colnames
                             dtf03[,c(1,2,4,3)]}))
        dt_metrics_norm
        }
        } ) )

fn <- dt_metrics_year_norm$STATION[1]

lst_metrics <- fn_aggregate_metrics(in_wd,fn,dt_metrics_year_norm)
save(lst_metrics,file=paste(fn,"_lst_metrics",".rds",sep=""))

return(1)
}

#Read in sensor files from tar files
#lst_years <- seq(1994,2024,1)
#xstation <- "74486094789"  #JFK
#xstation <- "74486454787"  #Farmingdale
#xstation <- "72502014734"  #Newark
#xstation <- "72502594741"  #Teterboro
#xstation <- "72505004781"  #MacArthur

#This reads in the tar files and then creates the R objects
#  72026723224_lst_metrics.rds
#  This contains the raw data in a list format
#  This is used as the starting point for creating the cvs files
#  Used to create the modeling matrix
library(data.table)

#Set the directory where the zipped tar files are located
wd <- "\\\\CXA01\\Users\\jhugh\\Documents\\HT\\NCEI_data"
#wd2 <- "\\\\CXA01\\Users\\jhugh\\Documents\\Py_S4\\Py_S4_v02_JHH\\NCEI_data"

setwd(wd)
lst_years <- seq(1994,2024,1)
dt_stations <- fread(file="Station_Info.csv")
## c("NY","MD", "VA","GA","KY","OH","TN")
## c("NH","VT","CT","MA","NJ","PA")
## c("GA","KY","TN","ME","RI","MT")
## c("ND", "SD", "NE", "KS", "FL")
## c("CA", "TX", "VA", "NC", "DE", "MD")

vc_states <- c("CA", "TX", "VA", "NC", "DE", "MD")
vc_stations <- dt_stations[US_State %in% vc_states, STATION]
#vc_stations <- vc_stations[211:310]

for(istn in vc_stations){
    check <- fn_create_norm_metrics_files(wd,lst_years,istn)
}

