library(data.table)
library(arrow)
library(woeBinning)
library(nnet)
library(caret)


#Get list of current s$ files
wd01 <- "~/Py_S4/NCEI_parquet_files/"
fpattern <- "*train.csv"
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


fn_mnl <- function(fnp){
  #fnp <- lst_data_files[10]
  print(fnp)
  dt <- fread(fnp)

  #colnames need to be updated as some old files did not have names correct
  ocolnames <- colnames(dt)
  ncolnames <- ocolnames
  ocolnames[1:10]
  str1 <- match('0',ncolnames) #tgt_bin
  str2 <- match('6',ncolnames) #hour_of_day
  str3 <- match('7',ncolnames) #month_of_year
  
  if(!is.na(str1)){ncolnames[str1] <- 'tgt_bin' }
  if(!is.na(str2)){ncolnames[str2] <- 'hour_of_day' }
  if(!is.na(str3)){ncolnames[str3] <- 'month_of_year' }

  colnames(dt) <- ncolnames
  
  dvn0 <- "tgt_bin"
  dvn1 <- "tgt_01"
  dtsub <- dt[,1:29]
  
  #Create binary target for woe
  dtsub[,eval(dvn1) := ifelse(get(dvn0) <= .5,0,1)]
  
  print("Binary target vs target")
  print(table(dtsub[[eval(dvn1)]],dtsub[[eval(dvn0)]]))
  
  #Convert months, hours to factors
  dtsub[,fct_hours := as.factor(get('hour_of_day'))]
  dtsub[,fct_months := as.factor(get('month_of_year'))]
  
  #Check for periods where no rain has fallen anywhere in region
  #in the past 8 hours
  nper = 8
  nn1 <- paste("all_stid_amt_rain","",sep="")
  nn2 <- paste("bin_stid_have_rain","",sep="")
  var_name <- c(paste("avg",1:nper,sep=""))
  dtsub[, eval(nn1) := rowSums(.SD), .SDcols = var_name]
  dtsub[, eval(nn2) := ifelse(get(nn1)>0,1,0)]
  
  print("Periods where no rain fell (0) within 100km for preceding 8 hours")
  print(table(dtsub[[eval(nn2)]]))
  print(table(dtsub[[eval(nn2)]],dtsub[[eval(dvn0)]]))

  #Convert counts to pct of stations with rain
  nper = 8
  var_create <- lapply(1:nper, function(x){
    nn1 <- paste("pct_stid_rain",x,sep="")
    on1 <- paste("cnt_gt_zero",x,sep="")
    on2 <- paste("cnt_ge_zero",x,sep="")
    dtsub[,eval(nn1) := 
            ifelse(get(on2) > 0,
                   get(on1)/
                     get(on2),NA)]
    return(1)
  })
  
  dtsub[,diff_avg1_2 := avg1 - avg2]
  dtsub[,diff_avg2_3 := avg2 - avg3]
  dtsub[,diff_avg3_4 := avg3 - avg4]
  dtsub[,diff_avg4_5 := avg4 - avg5]
  dtsub[,diff_avg5_6 := avg5 - avg6]
  dtsub[,diff_avg6_7 := avg6 - avg7]
  dtsub[,diff_avg7_8 := avg7 - avg8]

  #WoE analysis
  lst_ivar <- c("hour_of_day",
                "month_of_year",
                nn1,
                nn2,
                eval(paste("pct_stid_rain",1:8,sep="")),
                eval(paste("avg",c(1,2,3,4,5,6,7,8),sep="")),
                eval(paste("diff_avg",c("1_2","2_3","3_4","4_5","6_7","7_8"),sep="")))
  
  woe_bins <- woe.binning(dtsub,eval(dvn1),lst_ivar
                          ,min.perc.total=0.05
                          ,min.perc.class=0.001
                          ,stop.limit=0.1
  )
  woe_bins
  
  #Deploy bins
  df.with.binned.vars.added <- woe.binning.deploy(dtsub, woe_bins,
                                                  add.woe.or.dum.var='woe')

  #Results binomial model all records
  dep_var = dvn1
  new_var = colnames(df.with.binned.vars.added)
  ind_var <- new_var["woe" == substr(new_var,1,3)]
  
  reg_formula <- as.formula(paste(dep_var," ~ ", paste(ind_var, collapse="+")))
  
  dt <- na.omit(df.with.binned.vars.added)
  nmodel <- multinom(reg_formula, data = dt)
  
  pred_probs <- predict(nmodel,newdata=dt,type="probs")
  data_with_preds <- cbind(dt, pred_probs)
  
  rtn <- list(nmodel,data_with_preds)
  
  return(rtn)
}

#Get model output for all stations
lst_mnl_results <- lapply(lst_data_files, function(xfn) fn_mnl(xfn))

fn_confmat <- function(dtres,act_thresh,pred_thresh){
  
  dtres <- dtres[[2]]
  actual_values <- factor(dtres$tgt_bin >= act_thresh)
  predicted_values <- factor(dtres$pred_probs >= pred_thresh)
  cm <- confusionMatrix(data = predicted_values, 
                        reference = actual_values, 
                        positive = "TRUE")
  Precision <- cm$byClass[match("Precision",attr(cm$byClass,"names"))]
  Recall <- cm$byClass[match("Sensitivity",attr(cm$byClass,"names"))]
  attr(Recall,"names") <- "Recall"
  
  lst_rtn <- list(list(Precision,Recall),cm)
  return(lst_rtn)
  
}

fn_confmat_grid <- function(fname,dtres,vc_act,vc_pred){
  
  # iact <- vc_act[1]
  # iprd <- vc_pred[1]
  
  lst_rtn <- lapply(vc_act, function(xact) {
    Reduce(rbind,lapply(vc_pred, function(xprd) {
      res <- fn_confmat(dtres,xact,xprd)
      rtn <- data.table(t(res[[1]]))
  
      colnames(rtn) <- c("Precision","Recall")

      rtn$Actual_Threshold <- xact
      rtn$Prob_Threshold <- xprd  
      rtn$Source <- fname
      rtn
    }))
     
    })

  dt_rtn <- Reduce(rbind,lst_rtn)
  
  return(dt_rtn)
}


vc_act <- c(.1,.5,1,4)
vc_prb <- c(.1,.15,.2,.25,.3)

lst_grids <- lapply(1:length(lst_data_filename), function(xidx) {
  fn_confmat_grid(lst_data_filename[xidx],lst_mnl_results[[xidx]],vc_act,vc_prb)
})

dt_grid <- Reduce(rbind,lst_grids)
dt_grid[,fct_Prob_Threshold := factor(Prob_Threshold)]

dt_grid[,valueP := as.numeric(Precision)]
bx_Prec_med <- boxplot(valueP ~ fct_Prob_Threshold,data=dt_grid)
vc_Prec_med <- bx_Prec_med$stats[3,]

dt_grid[,valueR := as.numeric(Recall)]
bx_Recall_med <- boxplot(valueR ~ fct_Prob_Threshold,data=dt_grid)
vc_Recall_med <- bx_Recall_med$stats[3,]

xx <- vc_Prec_med*vc_Prec_med
yy <- vc_Recall_med*vc_Recall_med
zz <- sqrt(xx+yy)
zz

stids <- unique(dt_grid$Source)
stids
