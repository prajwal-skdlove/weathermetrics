#Examples of getting the data
library(data.table)

#Set the directory where the zipped tar files will be placed
setwd("\\\\CXA01\\Users\\jhugh\\Documents\\HT\\NCEI_data")
wd <- "\\\\CXA01\\Users\\jhugh\\Documents\\HT\\NCEI_data"
wd2 <- "\\\\CXA01\\Users\\jhugh\\Documents\\Py_S4\\Py_S4_v02_JHH\\NCEI_data"

#Get the files from NCEI website
  options(timeout = 3000)  # Set timeout to 3000 seconds (50 minutes)
  
  fn_get_file <- function(in_year){
  url <- paste('http://www.ncei.noaa.gov/data/global-hourly/archive/csv/',in_year,'.tar.gz',sep="")
  file <- basename(url)
  download.file(url, file)
  }
  
  lst_years <- seq(1994,2024,1)
  lapply(lst_years, function(x) fn_get_file(x))

