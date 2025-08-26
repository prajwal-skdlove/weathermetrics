library(data.table)
library(woeBinning)
library(nnet)

#fnp <- "~/Py_S4/Py_S4_v02_JHH/NCEI_parquet_files/74486094789_train.csv"
#fnp <- "~/Py_S4/Py_S4_v02_JHH/NCEI_parquet_files/74486094789_train.csv"
fnp <- "~/Py_S4/Py_S4_v02_JHH/NCEI_parquet_files/72502014734_train.csv"


dt <- fread(fnp)
dtsub <- dt[,1:29]

dtsub$tgt <- dtsub$`0`
dtsub$`0` <- NULL
dtsub[,tgt_bin := ifelse(tgt < .4,0,1)]
jj <- lapply(1:8, function(x){
  nn1 <- paste("pct_stid_rain",x,sep="")
  on1 <- paste("cnt_gt_zero",x,sep="")
  on2 <- paste("cnt_ge_zero",x,sep="")
  dtsub[,eval(nn1) := 
          ifelse(get(on2) > 0,
                get(on1)/
                get(on2),NA)]
})

dtsub[,tgt_bin := ifelse(tgt == 0,0,1)]

#WoE analysis
var = c(paste("avg",c(1,2,3,4,5,6,7,8),sep=""),
        paste("pct_stid_rain",c(1,2,3,4,5,6,7,8),sep=""))

woe_bins <- woe.binning(dtsub,"tgt_bin",var
                  ,min.perc.total=0.05
                  ,min.perc.class=0.001
                  ,stop.limit=0.1
                  )
woe_bins

#Deploy bins
df.with.binned.vars.added <- woe.binning.deploy(dtsub, woe_bins,
                                                add.woe.or.dum.var='woe')

dep_var = "tgt"
ind_var <- colnames(df.with.binned.vars.added)[seq(39,69,2)]
reg_formula <- as.formula(paste(dep_var," ~ ", paste(ind_var, collapse="+")))

nmodel <- multinom(reg_formula, data = df.with.binned.vars.added)

ypredict <- predict(nmodel)

yactual <- df.with.binned.vars.added$tgt

table(yactual,ypredict)

xx <- data.table(nmodel$fitted.values)
plot(as.numeric(unlist(xx[,7])),as.numeric(unlist(xx[,1])))
plot(as.numeric(unlist(xx[,7])),as.numeric(unlist(xx[,3])))


dep_var = "tgt_bin"
reg_formula <- as.formula(paste(dep_var," ~ ", paste(ind_var, collapse="+")))

nmodel <- multinom(reg_formula, data = df.with.binned.vars.added)

ypredict <- predict(nmodel)

yactual <- df.with.binned.vars.added$tgt

table(yactual,ypredict)

table(dtsub$tgt,dtsub$tgt_bin)
