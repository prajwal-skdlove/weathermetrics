
library(data.table)
library(arrow)
library(woeBinning)
library(nnet)
library(caret)

# Get list of files
fn_get_file_info_s4 <- function(ptrn){
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
  dt_infiles[,stid := substr(Fname,11,21)]
  return(dt_infiles)
}

dt_files <- fn_get_file_info_s4("*DAYSUM_bin1_5_20_Test")
lst_results <- lapply(1:dim(dt_files)[1], function(xf){
    res <- fn_conmat_s4(dt_files[xf])
    res})
length(lst_results)

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
lst_cm <- lapply(lst_results, function(xi) fn_summarize_confmat(xi))

dt_all <- Reduce(rbind,lst_cm)

