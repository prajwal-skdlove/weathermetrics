#install.packages("arrow")
library(data.table)
library(arrow)
library(woeBinning)
library(nnet)
library(caret)



#Get list of current s$ files
fn_get_file_info <- function(ptrn){
  wd01 <- "~/Py_S4/NCEI_parquet_files/"
  fpattern <- ptrn #"*DAYSUM_train.parquet"
  lst_data_filename <- list.files(wd01,pattern=fpattern,full.names =FALSE)
  lst_data_files <- list.files(wd01,pattern=fpattern,full.names =TRUE)
  lst_file_size <- as.vector(Reduce(rbind,as.numeric(lapply(lst_data_files, function(xf){
    mbs <- file.size(xf)/(1024*1024)
    mbs
  }))))
  
  dt_infiles <- data.table('Fname'=lst_data_filename,
                           'Fpath'=lst_data_files,
                           'Fsize'=lst_file_size)
  dt_infiles[,stid := substr(Fname,1,11)]
  dt_infiles_sub <- dt_infiles[Fsize >= 100,]
  return(dt_infiles_sub)
  #lst_mod_files <- dt_infiles_sub$Fpath
  #lst_mod_stid <- substr(dt_infiles_sub$Fname,1,11)
}

#Get data table of data file info
dt_files <- fn_get_file_info("*DAYSUM_bin1_5_20_train")


#Create bat file to run s4 models on same data files
lst_cmds <- unlist(lapply(2:dim(dt_files)[1], function(idx){
  #idx <- 1
  iepochs <- 100
  bat_content <- paste("python -m s4model --modelname",
                       paste("s4",dt_files$stid[idx],"DAYSUM_bin1_5_20",sep="_"),
                       "--modeltype classification --dataset ",
                       dt_files$Fpath[idx],
                       " --tabulardata --dependent_variable tgt_bin --epochs ",
                       iepochs,
                       sep=" ")
  bat_content
}))
file_conn <- file("C:\\Users\\jhugh\\Documents\\GitHub\\Py_Weather_S4\\weathermetrics\\s4model\\py_s4b_bin1_5_20.bat",open="w")
writeLines(lst_cmds,con=file_conn)
close(file_conn)




