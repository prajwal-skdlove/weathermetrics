library(data.table)
library(woeBinning)
library(nnet)

wd01 <- "~/Py_S4/Py_S4_v02_JHH/NCEI_parquet_files/"
stid <- "74486094789"  #JFK
stid <- "72278023183"  #Phoenix
stid <- "72202012839"  #Miami
stid <- "72243012960"  #Houston
stid <- "72502014734"  #Newark


postfix <- "_train.csv"
fnp <- paste(wd01,stid,postfix,sep="")
dt <- fread(fnp)

#Create a boolean target for WoE
dvn0 <- "tgt_bin"
dvn1 <- "tgt_01"
dtsub <- dt[,1:29]
dtsub[,eval(dvn1) := ifelse(get(dvn0) <= .5,0,1)]

table(dtsub[[eval(dvn1)]],dtsub[[eval(dvn0)]])

#Convert months, hours to factors
dtsub[,fct_hours := as.factor(get('6'))]
dtsub[,fct_months := as.factor(get('7'))]
table(dtsub$fct_months,dtsub$bin_stid_have_rain)

#Check for periods where no rain has fallen anywhere in region
#in the past 8 hours
nper = 8
nn1 <- paste("all_stid_amt_rain","",sep="")
nn2 <- paste("bin_stid_have_rain","",sep="")
var_name <- c(paste("avg",1:nper,sep=""))
dtsub[, eval(nn1) := rowSums(.SD), .SDcols = var_name]
dtsub[, eval(nn2) := ifelse(get(nn1)>0,1,0)]

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
lst_ivar <- c("6","7",
              nn1,nn2,
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
#vc <- c(2:4,5:29,32:39)
ind_var <- ind_var["woe" == substr(ind_var,1,3)]

reg_formula <- as.formula(paste(dep_var," ~ ", paste(ind_var, collapse="+")))

nmodel <- multinom(reg_formula, data = na.omit(df.with.binned.vars.added))
p2 <- nmodel$fitted.values

conf_mat <- as.matrix(table(na.omit(df.with.binned.vars.added)[[dvn0]],round(p2,1)))
conf_mat

recall_ge_30 <- as.vector(unlist(lapply(1:8, function(x) sum(conf_mat[x,4:6])/sum(conf_mat[x,1:6]))))
mres <- cbind(conf_mat,recall_ge_30)
mres

prec_ge_30 <- c(sum(conf_mat[4:8,4:6])/sum(conf_mat[,4:6]))


#Model with records out that have 8 periods no rain
#Results binomial model all records
dep_var = dvn1
#vc <- c(2:4,5:29,32:39)
ind_var <- ind_var["woe" == substr(ind_var,1,3)]

dt_model <- df.with.binned.vars.added[bin_stid_have_rain == 1,]
ind_var <- c(dep_var,ind_var)
dt_model <- dt_model[,..ind_var]
reg_formula <- as.formula(paste(dep_var," ~ .^2"))
nmodel <- multinom(reg_formula, data = na.omit(dt_model))
p2 <- nmodel$fitted.values

conf_mat <- as.matrix(table(na.omit(dt_model)[[dvn0]],round(p2,1)))
conf_mat

recall_ge_30 <- as.vector(unlist(lapply(1:8, function(x) sum(conf_mat[x,4:6])/sum(conf_mat[x,1:6]))))
mres <- cbind(conf_mat,recall_ge_30)
mres

prec_ge_30 <- c(sum(conf_mat[2:8,4:6])/sum(conf_mat[,4:6]))
prec_ge_30


#Model with predicted values
dt_model_pred <- dt_model[,predval := p2]

reg_formula <- as.formula(paste(paste(dep_var," ~ ", paste(ind_var, collapse="+"))," + predval",sep="")) 
nmodel <- multinom(reg_formula, data = na.omit(dt_model_pred))
p2 <- nmodel$fitted.values

conf_mat <- as.matrix(table(na.omit(dt_model)[[dvn0]],round(p2,1)))
conf_mat

recall_ge_30 <- as.vector(unlist(lapply(1:8, function(x) sum(conf_mat[x,4:7])/sum(conf_mat[x,1:7]))))
mres <- cbind(conf_mat,recall_ge_30)
mres

prec_ge_30 <- c(sum(conf_mat[2:8,4:7])/sum(conf_mat[,4:7]))
prec_ge_30



