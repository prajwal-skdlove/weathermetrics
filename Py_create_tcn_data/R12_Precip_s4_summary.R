
library(data.table)
library(arrow)
library(woeBinning)
library(nnet)
library(caret)

# Get list of files
fn_get_file_info_s4_Test <- function(ptrn){
  wd01 <- "C:\\Users\\jhugh\\Documents\\GitHub\\Py_Weather_S4\\weathermetrics\\results"
  fpattern <- ptrn
  lst_data_filename <- list.files(wd01,pattern=fpattern,full.names =FALSE)
  test_results <- grep("Test_results",lst_data_filename)
  lst_data_filename <- lst_data_filename[test_results]
  lst_data_files <- list.files(wd01,pattern=fpattern,full.names =TRUE)[test_results]
  lst_file_size <- as.vector(Reduce(rbind,as.numeric(lapply(lst_data_files, function(xf){
    mbs <- file.size(xf)/(1024*1024)
    mbs
  }))))
  
  dt_infiles <- data.table('Fname'=lst_data_filename,
                           'Fpath'=lst_data_files,
                           'Fsize'=lst_file_size)
  dt_infiles[,stid := substr(Fname,4,14)]
  dt_infiles[,ftype := "Test"]
  return(dt_infiles)
}

fn_get_file_info_s4_Valid <- function(ptrn){
  wd01 <- "C:\\Users\\jhugh\\Documents\\GitHub\\Py_Weather_S4\\weathermetrics\\results"
  fpattern <- ptrn
  lst_data_filename <- list.files(wd01,pattern=fpattern,full.names =FALSE)
  test_results <- grep("Validation_results",lst_data_filename)
  lst_data_filename <- lst_data_filename[test_results]
  lst_data_files <- list.files(wd01,pattern=fpattern,full.names =TRUE)[test_results]
  lst_file_size <- as.vector(Reduce(rbind,as.numeric(lapply(lst_data_files, function(xf){
    mbs <- file.size(xf)/(1024*1024)
    mbs
  }))))
  
  dt_infiles <- data.table('Fname'=lst_data_filename,
                           'Fpath'=lst_data_files,
                           'Fsize'=lst_file_size)
  dt_infiles[,stid := substr(Fname,4,14)]
  dt_infiles[,ftype := "Valid"]
  return(dt_infiles)
}

fn_get_file_info_s4_Train <- function(ptrn){
  wd01 <- "C:\\Users\\jhugh\\Documents\\GitHub\\Py_Weather_S4\\weathermetrics\\results"
  fpattern <- ptrn
  lst_data_filename <- list.files(wd01,pattern=fpattern,full.names =FALSE)
  train_results <- grep("Train_results",lst_data_filename)
  lst_data_filename <- lst_data_filename[train_results]
  lst_data_files <- list.files(wd01,pattern=fpattern,full.names =TRUE)[train_results]
  lst_file_size <- as.vector(Reduce(rbind,as.numeric(lapply(lst_data_files, function(xf){
    mbs <- file.size(xf)/(1024*1024)
    mbs
  }))))
  
  dt_infiles <- data.table('Fname'=lst_data_filename,
                           'Fpath'=lst_data_files,
                           'Fsize'=lst_file_size)
  dt_infiles[,stid := substr(Fname,4,14)]
  dt_infiles[,ftype := "Train"]
  return(dt_infiles)
}

# Check conf matrix for each model
fn_reorg_s4 <- function(ipf){
  infile <- ipf$Fpath
  stdid <- ipf$stid
  attr(stdid,"names") <- "Station ID"
  dt01 <- fread(infile)
  
  #tbl <- table(dt01$tgt_bin,dt01$Predicted)
  #names(dimnames(tbl)) <- list("Actual (row)","Predicted (col)")
  
  #dt01[,tgt_has_rain := factor(ifelse(tgt_bin == 0,FALSE,TRUE))]
  #dt01[,prd_has_rain := factor(ifelse(Predicted == 0,FALSE,TRUE))]
  ##print(table(dt01$tgt_has_rain,dt01$prd_has_rain))
  #cm <- confusionMatrix(data = dt01$tgt_has_rain, 
  #                      reference = dt01$prd_has_rain, 
  #                      positive = "TRUE")
  #Precision <- cm$byClass[match("Precision",attr(cm$byClass,"names"))]
  #Recall <- cm$byClass[match("Sensitivity",attr(cm$byClass,"names"))]
  #attr(Recall,"names") <- "Recall"
  
  #lst_rtn <- list(stdid,tbl,cm,list(Precision,Recall))
  
  return(dt01)
  
}

# Check conf matrix for each model
fn_conmat_s4 <- function(ipf){
  infile <- ipf$Fpath
  stdid <- ipf$stid
  attr(stdid,"names") <- "Station ID"
  dt01 <- fread(infile)
  
  tbl <- table(dt01$tgt_bin,dt01$Predicted)
  names(dimnames(tbl)) <- list("Actual (row)","Predicted (col)")
  
  dt01[,tgt_has_rain := factor(ifelse(tgt_bin == 0,FALSE,TRUE))]
  dt01[,prd_has_rain := factor(ifelse(Predicted == 0,FALSE,TRUE))]
  #print(table(dt01$tgt_has_rain,dt01$prd_has_rain))
  cm <- confusionMatrix(data = dt01$tgt_has_rain, 
                        reference = dt01$prd_has_rain, 
                        positive = "TRUE")
  Precision <- cm$byClass[match("Precision",attr(cm$byClass,"names"))]
  Recall <- cm$byClass[match("Sensitivity",attr(cm$byClass,"names"))]
  attr(Recall,"names") <- "Recall"
  
  lst_rtn <- list(stdid,tbl,cm,list(Precision,Recall))
  return(lst_rtn)
  
}

# Summarize conf matrix for across models
fn_summarize_confmat <- function(ix){
  #ix <- lst_results[[2]]
  dt3 <- data.table(cbind(attr(ix[[3]][3]$overall,"names"),as.numeric(ix[[3]][[3]])))
  colnames(dt3) <- c("Metric","Value")
  dt4 <- data.table(cbind(attr(ix[[3]][4]$byClass,"names"),as.numeric(ix[[3]][[4]])))
  colnames(dt4) <- c("Metric","Value")
  
  dt1 <- data.table(ix[[2]])
  dt1$key <- 1:dim(dt1)[1]
  
  dt2 <- data.table(ix[[3]][[2]])
  dt2$key <- 1:dim(dt2)[1]
  
  dt_metrics <- rbind(dt4,dt3)
  
  dt_metrics$key <- 1:dim(dt_metrics)[1]
  lst_dt <- list(dt1,dt2,dt_metrics)
  
  dt_res <- Reduce(function(x,y) merge(x,y,by="key",all.x=TRUE), lst_dt)
  dt_res$Station_ID <- ix[[1]]
  dt_res
}

dt_files_Test <- fn_get_file_info_s4_Test("*DAYSUM_bin1_5_20_Test")
dt_files_Train <- fn_get_file_info_s4_Train("*DAYSUM_bin1_5_20_Train")
dt_files_Valid <- fn_get_file_info_s4_Valid("*DAYSUM_bin1_5_20_Valid")

dt_files <- Reduce(rbind,list(dt_files_Test,dt_files_Train,dt_files_Valid))

#Identify the unique stations in the data
all_stid <- unique(dt_files$stid)

curr_stid <- all_stid[2]
curr_files <- dt_files[stid == curr_stid,]

#Select files to process for a station
xlist <- 1:dim(curr_files)[1]

lst_results <- lapply(xlist, function(xf){
    res <- fn_reorg_s4(curr_files[xf])
    res})
length(lst_results)

#Select single data file
lst_reorg <- lapply(lst_results,function(xf){
  curr_dt <- xf
  dt_outcome <- curr_dt[,1:2]
  dt_features <- curr_dt[,4:dim(curr_dt)[2]]
  dt_prob <- data.table(curr_dt[,3])
  dt_prob[, str_prob := substr(as.character(extra_features),2,nchar(as.character(extra_features))-1)]

  dt_prob[,pval_0 := as.vector(Reduce(cbind,lapply(1:dim(dt_prob)[1],function(xi) {
      ss <- strsplit(as.character(dt_prob[xi,2]),",")[[1]][1]
      ss <- as.numeric(strsplit(ss,":")[[1]][2])
      ss
      })))]

  dt_prob[,pval_1 := as.vector(Reduce(cbind,lapply(1:dim(dt_prob)[1],function(xi) {
      ss <- strsplit(as.character(dt_prob[xi,2]),",")[[1]][2]
      ss <- as.numeric(strsplit(ss,":")[[1]][2])
      ss
      })))]

  dt_prob[,pval_2 := as.vector(Reduce(cbind,lapply(1:dim(dt_prob)[1],function(xi) {
      ss <- strsplit(as.character(dt_prob[xi,2]),",")[[1]][3]
      ss <- as.numeric(strsplit(ss,":")[[1]][2])
      ss
      })))]

  dt_prob[,pval_3 := as.vector(Reduce(cbind,lapply(1:dim(dt_prob)[1],function(xi) {
      ss <- strsplit(as.character(dt_prob[xi,2]),",")[[1]][4]
      ss <- as.numeric(strsplit(ss,":")[[1]][2])
      ss
      })))]

  dt_prob <- dt_prob[,3:6]

  dt_output <- cbind(dt_outcome,dt_prob)
  dt_output <- cbind(dt_output,dt_features)

  dt_output
})
length(lst_reorg)

#Concatenate the three data sets
all_data <- Reduce(rbind,lst_reorg)
rm(lst_reorg)

oname <- curr_files$Fpath[1]
oname <- gsub(".csv","_recomb.parquet",oname)
oname <- gsub("Test_results","results",oname)
oname

#Save the file as a parquet file

write_parquet(all_data, oname)

