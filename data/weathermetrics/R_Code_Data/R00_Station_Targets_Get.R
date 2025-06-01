
#Identify nearby stations and check if they have metric files
library(data.table)

ipd01 <- "\\\\CXA01\\Users\\jhugh\\Documents\\HT\\NCEI_data\\metrics_csv"
ipd02 <- "\\\\CXA01\\Users\\jhugh\\Documents\\HT\\NCEI_data\\metrics_csv_temp_store"

in_file <- paste(ipd01,"\\","Target_station_analysis_set",".csv",sep="") 
dt_station_data <- fread(in_file)

for (i in 1:450){
    inf <-  paste(dt_station_data[i,path],dt_station_data[i,fnames],sep="\\") 
    onf <- paste(ipd02,dt_station_data[i,fnames],sep="\\")
    file.copy(from = inf, to = onf)
}






