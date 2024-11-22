# Plot aggregated results ##############################################################
# import libraries
library(ggplot2)
library(ggtext)
library(dplyr)
library(tidyr)
library(forcats)
library(patchwork)

# Plot Theme
fontsize = 18
fontsize_ticks_legends = 16
theme_set(theme_minimal(base_size = fontsize) +
          theme(legend.title = element_text(size = fontsize_ticks_legends),
                legend.text = element_text(size = fontsize_ticks_legends),
                axis.title = element_text(size = fontsize_ticks_legends),
                axis.text = ggtext::element_markdown(size = fontsize_ticks_legends),
                strip.text = element_text(size = fontsize_ticks_legends),
                plot.caption = ggtext::element_markdown()))

DIR = "results/hyper_params_ablation"
# change the FILTER to the hyperparameter ablation you want to plot
FILTER = "_trust"
# Load results
results = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))

# Load DE baseline
results_de = read.csv(paste0(DIR, "/aggr_results_de.csv"))
results <- results |>
    select(-de_time, -de_lppd, -de_rmse) |>
    left_join(
        results_de |>
            filter(hidden_structure %in% unique(results$hidden_structure)) |>
            select(rng, de_time, de_lppd, de_rmse),
        by = "rng"
    )

# helper fun
custom_labels <- function(x, bold_label) {
  sapply(x, function(label) {
    formatted_label <- ifelse(label >= 1000, paste0(label / 1000, "k"), as.character(label))
    if (label == bold_label) {
      return(paste0("<b>", formatted_label, "</b>"))
    } else {
      return(as.character(formatted_label))
    }
  })
}



# Hyperparameter Ablations #############################################################
# NUTS baseline
DIR = "results/hyper_params_ablation"
FILTER = "_nuts"

nuts_baseline <- read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
nuts_lppd <- mean(nuts_baseline$lppd)
nuts_rmse <- mean(nuts_baseline$rmse)

# Desired Energy Variance ----------------------------
results |>
    group_by(data, desired_energy_var_start, desired_energy_var_end) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    arrange(desc(mean))

results |>
    group_by(desired_energy_var_start, desired_energy_var_end) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    mutate(method = paste0(desired_energy_var_start, " - ", desired_energy_var_end)) |>
    ggplot(aes(x = fct_reorder(method, mean), y = mean)) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2) +
    labs(x = "Energy Variance", y = "LPPD") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

plt_data <- results |>
    group_by(desired_energy_var_start, desired_energy_var_end) |>
    summarise(
        sd_lppd = sd(lppd),
        sd_rmse = sd(rmse),
        sd_de_lppd = sd(de_lppd),
        sd_de_rmse = sd(de_rmse),
        sd_L = sd(L),
        sd_step_size = sd(step_size),
        lppd = mean(lppd),
        rmse = mean(rmse),
        de_lppd = mean(de_lppd),
        de_rmse = mean(de_rmse),
        L = mean(L),
        step_size = mean(step_size)
    ) |>
    ungroup() |>
    pivot_longer(
        cols = c("lppd", "rmse", "L", "step_size", "sd_lppd", "sd_rmse", "sd_L", "sd_step_size", "sd_de_lppd", "sd_de_rmse", "de_lppd", "de_rmse"),
        names_to = "metric",
        values_to = "value") |>
    mutate(method = paste0(desired_energy_var_start, " - ", desired_energy_var_end))

plt_data <- plt_data |>
    filter(metric %in% c("lppd", "rmse", "L", "step_size")) |>
    bind_cols(
        plt_data |>
            filter(metric %in% c("sd_lppd", "sd_rmse", "sd_L", "sd_step_size")) |>
            rename(sd_value = value) |>
            select(sd_value)
    ) |>
    mutate(
        metric = case_when(
            metric == "lppd" ~ "LPPD",
            metric == "rmse" ~ "RMSE",
            metric == "L" ~ "L",
            metric == "step_size" ~ "Step Size"
        ),
        hline = case_when(
            metric == "RMSE" ~ nuts_rmse,
            metric == "LPPD" ~ nuts_lppd,
            TRUE ~ NA_real_  # No horizontal line for other metrics
        ),
        hline_de = case_when(
            metric == "LPPD" ~ mean(plt_data |>
                                      filter(metric == "de_lppd") |>
                                      pull(value)),
            metric == "RMSE" ~ mean(plt_data |>
                                    filter(metric == "de_rmse") |>
                                    pull(value)),
            TRUE ~ NA_real_  # No horizontal line for other metrics
        ),
        metric = factor(metric, levels = c("LPPD", "RMSE", "L", "Step Size"))
    )

plt <- plt_data |>
    ggplot(aes(x = method, y = value)) +
    geom_line(aes(group = metric), color = "#F35B04") +
    geom_hline(data = filter(plt_data, !is.na(hline)), aes(yintercept = hline),
               linetype = "dashed", color = "#3D348B", size = 1) +
    geom_hline(data = filter(plt_data, !is.na(hline_de)), aes(yintercept = hline_de),
                linetype = "dotted", color = "#014704", size = 1) +
    geom_point(size = 2, color = "#F35B04") +
    geom_errorbar(aes(ymin = value - sd_value, ymax = value + sd_value), width = 0.0, color = "#F35B04") +
    labs(
        x = "Energy Variance Schedule (From - To)", y = "",
        caption = "<span style='color:#014704'>avg. DE (dotted)</span><br><span style='color:#3D348B'>avg. BDE (dashed)</span>"
    ) +
    facet_wrap(~metric, scales = "free_y", switch = "y") +
    scale_x_discrete(labels = function(x) {custom_labels(x, "0.5 - 0.1")}) +
    theme(
        legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1, size = fontsize_ticks_legends-2),
        strip.placement = "outside", 
        axis.title.y = element_blank() 
    )


plt

ggsave(paste0(DIR, "/desired_energy_variance", FILTER, ".pdf"), plt, width = 7, height = 5)

# Effective Sample Size ----------------------------
results |>
    group_by(data, num_effective_samples) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    arrange(desc(mean))

results |>
    group_by(num_effective_samples) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    ggplot(aes(x = num_effective_samples, y = mean)) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2) +
    labs(x = "Effective Sample Size", y = "LPPD")

plt_data <- results |>
    group_by(num_effective_samples) |>
    summarise(
        sd_lppd = sd(lppd),
        sd_rmse = sd(rmse),
        sd_de_lppd = sd(de_lppd),
        sd_de_rmse = sd(de_rmse),
        sd_L = sd(L),
        sd_step_size = sd(step_size),
        lppd = mean(lppd),
        rmse = mean(rmse),
        de_lppd = mean(de_lppd),
        de_rmse = mean(de_rmse),
        L = mean(L),
        step_size = mean(step_size)
    ) |>
    pivot_longer(
        cols = c("lppd", "rmse", "L", "step_size", "sd_lppd", "sd_rmse", "sd_L", "sd_step_size", "sd_de_lppd", "sd_de_rmse", "de_lppd", "de_rmse"),
        names_to = "metric",
        values_to = "value")

plt_data <- plt_data |>
    filter(metric %in% c("lppd", "rmse", "L", "step_size")) |>
    bind_cols(
        plt_data |>
            filter(metric %in% c("sd_lppd", "sd_rmse", "sd_L", "sd_step_size")) |>
            rename(sd_value = value) |>
            select(sd_value)
    ) |>
    mutate(
        metric = case_when(
            metric == "lppd" ~ "LPPD",
            metric == "rmse" ~ "RMSE",
            metric == "L" ~ "L",
            metric == "step_size" ~ "Step Size"
        ),
        hline = case_when(
            metric == "RMSE" ~ nuts_rmse,
            metric == "LPPD" ~ nuts_lppd,
            TRUE ~ NA_real_  # No horizontal line for other metrics
        ),
        hline_de = case_when(
            metric == "LPPD" ~ mean(plt_data |>
                                      filter(metric == "de_lppd") |>
                                      pull(value)),
            metric == "RMSE" ~ mean(plt_data |>
                                    filter(metric == "de_rmse") |>
                                    pull(value)),
            TRUE ~ NA_real_  # No horizontal line for other metrics
        ),
        metric = factor(metric, levels = c("LPPD", "RMSE", "L", "Step Size"))
    )
plt <- plt_data |>
    ggplot(aes(x = num_effective_samples, y = value)) +
    geom_line(color = "#F35B04") +
    geom_hline(data = filter(plt_data, !is.na(hline)), aes(yintercept = hline),
               linetype = "dashed", color = "#3D348B", size = 1) +
    geom_hline(data = filter(plt_data, !is.na(hline_de)), aes(yintercept = hline_de),
                linetype = "dotted", color = "#014704", size = 1) +
    geom_point(size = 2, color = "#F35B04") +
    geom_errorbar(aes(ymin = value - sd_value, ymax = value + sd_value), width = 0.0, color = "#F35B04") +
    labs(
        x = "Effective Samples", y = "",
        caption = "<span style='color:#014704'>avg. DE (dotted)</span><br><span style='color:#3D348B'>avg. BDE (dashed)</span>"
    ) +
    scale_x_continuous(
        breaks = unique(plt_data$num_effective_samples),
        labels = function(x) custom_labels(x, 100)
    ) +
    facet_wrap(~metric, scales = "free_y", switch = "y") +
    theme(
        legend.position = "none",
        strip.placement = "outside", 
        axis.title.y = element_blank() 
    )
plt

ggsave(paste0(DIR, "/effective_samples", FILTER, ".pdf"), plt, width = 7, height = 5)

# Warmstart Budget ----------------------------
results |>
    group_by(data, warmup_steps) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    arrange(desc(mean))

results |>
    group_by(warmup_steps) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    ggplot(aes(x = warmup_steps, y = mean)) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2) +
    labs(x = "Warmstart Budget", y = "LPPD")

plt_data <- results |>
    group_by(warmup_steps) |>
    summarise(
        sd_lppd = sd(lppd),
        sd_rmse = sd(rmse),
        sd_de_lppd = sd(de_lppd),
        sd_de_rmse = sd(de_rmse),
        sd_L = sd(L),
        sd_step_size = sd(step_size),
        lppd = mean(lppd),
        rmse = mean(rmse),
        de_lppd = mean(de_lppd),
        de_rmse = mean(de_rmse),
        L = mean(L),
        step_size = mean(step_size)
    ) |>
    pivot_longer(
        cols = c("lppd", "rmse", "L", "step_size", "sd_lppd", "sd_rmse", "sd_L", "sd_step_size", "sd_de_lppd", "sd_de_rmse", "de_lppd", "de_rmse"),
        names_to = "metric",
        values_to = "value")

plt_data <- plt_data |>
    filter(metric %in% c("lppd", "rmse", "L", "step_size")) |>
    bind_cols(
        plt_data |>
            filter(metric %in% c("sd_lppd", "sd_rmse", "sd_L", "sd_step_size")) |>
            rename(sd_value = value) |>
            select(sd_value)
    ) |>
    mutate(
        metric = case_when(
            metric == "lppd" ~ "LPPD",
            metric == "rmse" ~ "RMSE",
            metric == "L" ~ "L",
            metric == "step_size" ~ "Step Size"
        ),
        hline = case_when(
            metric == "RMSE" ~ nuts_rmse,
            metric == "LPPD" ~ nuts_lppd,
            TRUE ~ NA_real_  # No horizontal line for other metrics
        ),
        hline_de = case_when(
            metric == "LPPD" ~ mean(plt_data |>
                                      filter(metric == "de_lppd") |>
                                      pull(value)),
            metric == "RMSE" ~ mean(plt_data |>
                                    filter(metric == "de_rmse") |>
                                    pull(value)),
            TRUE ~ NA_real_  # No horizontal line for other metrics
        ),
        metric = factor(metric, levels = c("LPPD", "RMSE", "L", "Step Size"))
    )
plt <- plt_data |>
    ggplot(aes(x = warmup_steps, y = value)) +
    geom_line(color = "#F35B04") +
    geom_hline(data = filter(plt_data, !is.na(hline)), aes(yintercept = hline),
               linetype = "dashed", color = "#3D348B", size = 1) +
    geom_hline(data = filter(plt_data, !is.na(hline_de)), aes(yintercept = hline_de),
                linetype = "dotted", color = "#014704", size = 1) +
    geom_point(size = 2, color = "#F35B04") +
    geom_errorbar(aes(ymin = value - sd_value, ymax = value + sd_value), width = 0.0, color = "#F35B04") +
    labs(
        x = "Warmup Steps/Budget", y = "",
        caption = "<span style='color:#014704'>avg. DE (dotted)</span><br><span style='color:#3D348B'>avg. BDE (dashed)</span>"
    ) +
    facet_wrap(~metric, scales = "free_y", switch = "y") +
    scale_x_continuous(
        breaks = unique(plt_data$warmup_steps),
        labels = function(x) custom_labels(x, 50000)
    ) +
    theme(
        legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1, size = fontsize_ticks_legends-1),
        strip.placement = "outside", 
        axis.title.y = element_blank() 
    )
plt

ggsave(paste0(DIR, "/warmstart_budget", FILTER, ".pdf"), plt, width = 7, height = 5)

# Trust in Estimate ----------------------------
results |>
    group_by(data, trust_in_estimate) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    arrange(desc(mean))

results |>
    group_by(trust_in_estimate) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    ggplot(aes(x = trust_in_estimate, y = mean)) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2) +
    labs(x = "Trust in Estimate", y = "LPPD")

plt_data <- results |>
    group_by(trust_in_estimate) |>
    summarise(
        sd_lppd = sd(lppd),
        sd_rmse = sd(rmse),
        sd_de_lppd = sd(de_lppd),
        sd_de_rmse = sd(de_rmse),
        sd_L = sd(L),
        sd_step_size = sd(step_size),
        lppd = mean(lppd),
        rmse = mean(rmse),
        de_lppd = mean(de_lppd),
        de_rmse = mean(de_rmse),
        L = mean(L),
        step_size = mean(step_size)
    ) |>
    pivot_longer(
        cols = c("lppd", "rmse", "L", "step_size", "sd_lppd", "sd_rmse", "sd_L", "sd_step_size", "sd_de_lppd", "sd_de_rmse", "de_lppd", "de_rmse"),
        names_to = "metric",
        values_to = "value")

plt_data <- plt_data |>
    filter(metric %in% c("lppd", "rmse", "L", "step_size")) |>
    bind_cols(
        plt_data |>
            filter(metric %in% c("sd_lppd", "sd_rmse", "sd_L", "sd_step_size")) |>
            rename(sd_value = value) |>
            select(sd_value)
    ) |>
    mutate(
        metric = case_when(
            metric == "lppd" ~ "LPPD",
            metric == "rmse" ~ "RMSE",
            metric == "L" ~ "L",
            metric == "step_size" ~ "Step Size"
        ),
        hline = case_when(
            metric == "RMSE" ~ nuts_rmse,
            metric == "LPPD" ~ nuts_lppd,
            TRUE ~ NA_real_  # No horizontal line for other metrics
        ),
        hline_de = case_when(
            metric == "LPPD" ~ mean(plt_data |>
                                      filter(metric == "de_lppd") |>
                                      pull(value)),
            metric == "RMSE" ~ mean(plt_data |>
                                    filter(metric == "de_rmse") |>
                                    pull(value)),
            TRUE ~ NA_real_  # No horizontal line for other metrics
        ),
        metric = factor(metric, levels = c("LPPD", "RMSE", "L", "Step Size"))
    )

plt <- plt_data |>
    ggplot(aes(x = trust_in_estimate, y = value)) +
    geom_line(color = "#F35B04") +
    geom_hline(data = filter(plt_data, !is.na(hline)), aes(yintercept = hline),
               linetype = "dashed", color = "#3D348B", size = 1) +
    geom_hline(data = filter(plt_data, !is.na(hline_de)), aes(yintercept = hline_de),
                linetype = "dotted", color = "#014704", size = 1) +
    geom_point(size = 2, color = "#F35B04") +
    geom_errorbar(aes(ymin = value - sd_value, ymax = value + sd_value), width = 0.0, color = "#F35B04") +
    labs(
        x = "Trust in Estimate", y = "",
        caption = "<span style='color:#014704'>avg. DE (dotted)</span><br><span style='color:#3D348B'>avg. BDE (dashed)</span>"
    ) +
    scale_x_continuous(
        breaks = unique(plt_data$trust_in_estimate),
        labels = function(x) custom_labels(x, 1.5)
    ) +
    facet_wrap(~metric, scales = "free_y", switch = "y") +
    theme(
        legend.position = "none",
        strip.placement = "outside", 
        axis.title.y = element_blank()
    )

plt

ggsave(paste0(DIR, "/trust_in_estimate", FILTER, ".pdf"), plt, width = 7, height = 5)



# Complexity Ablations ################################################################
DIR = "results/complexity_ablation"
FILTER = "_mclmc"
results_mclmc = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
FILTER = "_nuts"
results_nuts = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
FILTER = "_de"
results_de = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
results = results_mclmc |>
    mutate(sampler = "MCLMC") |>
    bind_rows(results_nuts |>
              mutate(sampler = "NUTS"))

architecture <- c(
    "8-8-8-2",
    "16-16-16-2",
    "32-32-32-2",
    "48-48-48-2"
)
n_params <- c(266, 786, 2594, 5426)

convert_architecture <- function(arch, params) {
    arch <- strsplit(arch, "-")[[1]]
    n_layers <- length(arch) - 1
    out <- paste0(
        as.character(n_layers),
        "x",
        arch[1],
        " (", params, ")"
    )
    out
}
parsed_architecture <- mapply(convert_architecture, architecture, n_params)
results$hidden_structure <- factor(
    results$hidden_structure,
    levels = architecture,
    labels = parsed_architecture
)

results |>
    group_by(sampler, hidden_structure) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    arrange(desc(mean))

results <- results |>
    mutate(
        sampling_time = sampling_time / 60,
        de_time = de_time / 60,
        ratio = sampling_time / de_time
    )

results_de <- results_de |>
    mutate(
        hidden_structure = factor(
            hidden_structure,
            levels = architecture,
            labels = parsed_architecture
        )
    )

results_de <- results_de |>
    mutate(lppd = de_lppd, rmse = de_rmse) |>
    mutate(sampler = "DE") |>
    select(sampler, hidden_structure, lppd, rmse)
results <- results |>
    bind_rows(results_de)

plt_data_main <- results |>
    group_by(sampler, hidden_structure) |>
    summarise(
        sd_lppd = sd(lppd),
        sd_rmse = sd(rmse),
        sd_de_lppd = sd(de_lppd),
        sd_de_rmse = sd(de_rmse),
        sd_L = sd(L),
        sd_step_size = sd(step_size),
        sd_sampling_time = sd(sampling_time),
        sd_ratio = sd(ratio),
        lppd = mean(lppd),
        rmse = mean(rmse),
        de_lppd = mean(de_lppd),
        de_rmse = mean(de_rmse),
        L = mean(L),
        step_size = mean(step_size),
        sampling_time = mean(sampling_time),
        ratio = mean(ratio)
    ) |>
    pivot_longer(
        cols = c("lppd", "rmse", "L", "step_size", "sampling_time", "ratio", "de_lppd", "de_rmse",
                 "sd_lppd", "sd_rmse", "sd_L", "sd_step_size", "sd_sampling_time", "sd_ratio", "sd_de_lppd", "sd_de_rmse"),
        names_to = "metric",
        values_to = "value")

plt_data <- plt_data_main |>
    filter(metric %in% c("lppd", "rmse", "sampling_time", "ratio")) |>
    ungroup() |>
    bind_cols(
        plt_data_main |>
            filter(metric %in% c("sd_lppd", "sd_rmse", "sd_sampling_time", "sd_ratio")) |>
            rename(sd_value = value) |>
            ungroup() |>
            select(sd_value)
    ) |>
    mutate(
        metric = case_when(
            metric == "lppd" ~ "LPPD",
            metric == "rmse" ~ "RMSE",
            metric == "sampling_time" ~ "Sampling Time (min)",
            metric == "ratio" ~ "Sampling / Optimization Time"
        ),
        metric = factor(metric, levels = c("Sampling Time (min)", "Sampling / Optimization Time", "LPPD", "RMSE")
        ),
    )

plt <- plt_data |>
    filter(metric %in% c("LPPD", "RMSE")) |>
    mutate(sampler = case_when(
        sampler == "MCLMC" ~ "MILE",
        sampler == "NUTS" ~ "BDE",
        TRUE ~ sampler
    )) |>
    mutate(sampler = factor(sampler, levels = c("MILE", "BDE", "DE"))) |>
    ggplot(aes(x = hidden_structure, y = value, color = sampler)) +
    geom_line(aes(group = sampler)) +
    geom_point(size = 2) +
    geom_ribbon(
        aes(
            ymin = value - sd_value,
            ymax = value + sd_value,
            group=sampler, fill=sampler, color=NULL
        ),
        alpha = 0.2
    ) +
    labs(
        x = "Hidden Layers x Neurons (Parameters)", y = "",
        color = "", fill = "",
    ) +
    scale_color_manual(values = c("MILE" = "#F35B04", "BDE" = "#3D348B", "DE" = "grey")) +
    scale_fill_manual(values = c("MILE" = "#F35B04", "BDE" = "#3D348B", "DE" = "grey")) +
    facet_wrap(~metric, scales = "free_y") +
    theme(legend.position = "bottom")

perf_for_complexity_plots <- plt
plt

ggsave(paste0(DIR, "/hidden_structure", FILTER, ".pdf"), plt, width = 12, height = 6)

# just the sampling times
sampling_times <- plt_data |>
    filter(metric == "Sampling Time (min)") |>
    filter(sampler != "DE") |>
    mutate(hidden_structure = factor(
        hidden_structure,
        levels = parsed_architecture,
        labels = n_params
    ),
    params = as.numeric(stringr::str_extract(hidden_structure, "\\d+")))

# fit lm to the sampling times
lm_sampling_times_mclmc_comp <- lm(log(value) ~ log(params), data = sampling_times |> filter(sampler == "MCLMC"))
summary(lm_sampling_times_mclmc_comp)
lm_sampling_times_nuts_comp <- lm(log(value) ~ log(params), data = sampling_times |> filter(sampler == "NUTS"))
summary(lm_sampling_times_nuts_comp)


# fit a model directly to the ratio
ratio_df <- sampling_times |>
    filter(sampler == "NUTS") |>
    mutate(ratio = value / sampling_times |> filter(sampler == "MCLMC") |> pull(value))

lm_ratio <- lm(ratio ~ (params), data = ratio_df)
plot(log(ratio_df$params), (ratio_df$ratio))
ratio_df |>
    ggplot(aes(x = params, y = ratio)) +
    geom_point() +
    geom_smooth(method = "lm", se = T) +
    labs(x = "Number of Parameters", y = "Sampling Time Ratio (NUTS / MCLMC)") +
    scale_x_log10() +
    # aspect ratio
    theme(aspect.ratio = 8/10)
summary(lm_ratio)


sampling_times |>
    ggplot(aes(x = params, y = value, color = sampler)) +
    geom_line(aes(group = sampler)) +
    geom_point(size = 2) +
    geom_ribbon(
        aes(
            ymin = value - sd_value,
            ymax = value + sd_value,
            group=sampler, fill=sampler, color=NULL
        ),
        alpha = 0.2
    ) +
    scale_x_log10() +
    scale_y_log10() +
    labs(
        x = "Number of Parameters (log-scale)", y = "Sampling Time (log-scale)",
        color = "", fill = "",
        title = "Sampling Time Scaling in Parameters",
    ) +
    scale_x_continuous(
        breaks = unique(sampling_times$params),
        labels = unique(sampling_times$hidden_structure),
        transform = "log10"
    ) +
    scale_color_manual(values = c("MCLMC" = "#F35B04", "NUTS" = "#3D348B", "DE" = "grey")) +
    scale_fill_manual(values = c("MCLMC" = "#F35B04", "NUTS" = "#3D348B", "DE" = "grey")) +
    theme(
        plot.title = element_text(size = 18, hjust = 0.5),
        legend.position = "none"
    )

sampling_plt <- sampling_times |>
    ggplot(aes(x = params, y = value, color = sampler)) +
    geom_line(aes(group = sampler)) +
    geom_point(size = 2) +
    geom_ribbon(
        aes(
            ymin = value - sd_value,
            ymax = value + sd_value,
            group=sampler, fill=sampler, color=NULL
        ),
        alpha = 0.2
    ) +
    labs(
        x = "Number of Parameters", y = "Sampling Time (min)",
        color = "", fill = "",
        title = "Sampling Time Scaling in Parameters",
    ) +
    geom_function(fun = function(x) exp(predict(lm_sampling_times_mclmc_comp, newdata = data.frame(params = (x)))), color = "#F35B04", linetype = "dashed") +
    geom_function(fun = function(x) exp(predict(lm_sampling_times_nuts_comp, newdata = data.frame(params = (x)))), color = "#3D348B", linetype = "dashed") +
    scale_x_continuous(
        breaks = unique(sampling_times$params),
        labels = unique(sampling_times$hidden_structure)
    ) +
    scale_color_manual(values = c("MCLMC" = "#F35B04", "NUTS" = "#3D348B", "DE" = "grey")) +
    scale_fill_manual(values = c("MCLMC" = "#F35B04", "NUTS" = "#3D348B", "DE" = "grey")) +
    theme(
        plot.title = element_text(size = 18, hjust = 0.5),
        legend.position = "none"
    )
sampling_plt_params <- sampling_plt
sampling_plt
ggplot2::ggsave(paste0(DIR, "/sampling_times", FILTER, ".pdf"), sampling_plt, width = 8, height = 6)

# Datasize Ablations ###################################################################
DIR = "results/datasize_ablation"
FILTER = "_mclmc"
results_mclmc = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
FILTER = "_nuts"
results_nuts = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
results <- results_mclmc |>
    mutate(sampler = "MCLMC") |>
    bind_rows(results_nuts |>
              mutate(sampler = "NUTS"))

results |>
    group_by(sampler, datapoint_limit) |>
    summarize(mean = mean(lppd), sd = sd(lppd)) |>
    arrange(desc(mean))

# just the sampling times
sampling_times <- results |>
    mutate(observations = datapoint_limit, sampling_time = sampling_time / 3600) |>
    group_by(sampler, observations) |>
    summarize(mean_time = mean(sampling_time), sd_time = sd(sampling_time))
lm_sampling_times_mclmc <- lm(mean_time ~ I(observations^2), data = sampling_times |> filter(sampler == "MCLMC"))
summary(lm_sampling_times_mclmc)
lm_sampling_times_nuts <- lm(log(mean_time) ~ observations, data = sampling_times |> filter(sampler == "NUTS"))
summary(lm_sampling_times_nuts)
lm_sampling_times_nuts <- lm(mean_time ~ I(observations^2), data = sampling_times |> filter(sampler == "NUTS"))
summary(lm_sampling_times_nuts)

ratio <- summary(lm_sampling_times_nuts)$coefficients[2, 1] / summary(lm_sampling_times_mclmc)$coefficients[2, 1]
ratio
# 6.6
# ratio directly
ratio_df <- sampling_times |>
    filter(sampler == "NUTS") |>
    mutate(ratio = mean_time / sampling_times |> filter(sampler == "MCLMC") |> pull(mean_time))

lm_ratio <- lm(ratio ~ (observations), data = ratio_df)
plot(ratio_df$observations, ratio_df$ratio)
summary(lm_ratio)

ratio_df |>
    ggplot(aes(x = observations, y = ratio)) +
    geom_point() +
    geom_smooth(method = "lm", se = T) +
    labs(x = "Number of Observations", y = "Sampling Time Ratio (NUTS / MCLMC)") +
    # aspect ratio
    theme(aspect.ratio = 8/10)

sampling_times |>
    ggplot(aes(x = observations, y = mean_time, color = sampler)) +
    geom_line(aes(group = sampler)) +
    geom_point(size = 2) +
    geom_ribbon(
        aes(
            ymin = mean_time - sd_time,
            ymax = mean_time + sd_time,
            group=sampler, fill=sampler, color=NULL
        ),
        alpha = 0.2
    ) +
    scale_x_log10() +
    scale_y_log10() +
    labs(
        x = "Number of Observations (log-scale)", y = "Sampling Time (log-scale)",
        color = "", fill = "",
        title = "Sampling Time Scaling in Observations",
    ) +
    scale_color_manual(values = c("MCLMC" = "#F35B04", "NUTS" = "#3D348B")) +
    scale_fill_manual(values = c("MCLMC" = "#F35B04", "NUTS" = "#3D348B")) +
    theme(
        plot.title = element_text(size = 18, hjust = 0.5),
        legend.position = "none"
    )

sampling_plt <- sampling_times |>
    ggplot(aes(x = observations, y = mean_time, color = sampler)) +
    geom_line(aes(group = sampler)) +
    geom_point(size = 2) +
    geom_ribbon(
        aes(
            ymin = mean_time - sd_time,
            ymax = mean_time + sd_time,
            group=sampler, fill=sampler, color=NULL
        ),
        alpha = 0.2
    ) +
    labs(
        x = "Number of Observations", y = "Sampling Time (h)",
        color = "", fill = "",
        title = "Sampling Time Scaling in Observations",
    ) +
    geom_function(
        fun = function(x) coef(lm_sampling_times_mclmc)[1] + coef(lm_sampling_times_mclmc)[2] * x^2,
        color = "#F35B04", linetype = "dashed"
    ) +
    geom_function(
        fun = function(x) coef(lm_sampling_times_nuts)[1] + coef(lm_sampling_times_nuts)[2] * x^2,
        color = "#3D348B", linetype = "dashed"
    ) +
    scale_color_manual(values = c("MCLMC" = "#F35B04", "NUTS" = "#3D348B")) +
    scale_fill_manual(values = c("MCLMC" = "#F35B04", "NUTS" = "#3D348B")) +
    theme(
        plot.title = element_text(size = 18, hjust = 0.5),
        legend.position = "none"
    )
sampling_plt_data <- sampling_plt
sampling_plt

ggsave(paste0(DIR, "/datasize", FILTER, ".pdf"), sampling_plt, width = 8, height = 6)

# combined sampling times plot (use patchwork)
combined_scaling_plot <- (sampling_plt_params + sampling_plt_data) / perf_for_complexity_plots
combined_scaling_plot

ggplot2::ggsave(paste0(DIR, "/combined_scaling_plot", FILTER, ".pdf"), combined_scaling_plot, width = 13, height = 10)

# UCI Replication ######################################################################
DIR = "results/repl_uci"
# FILTER = "_de"
# FILTER = "_nuts"
FILTER = "_mclmc"
# Load results
results = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
results

results |>
    group_by(data) |>
    summarize(
        bde_lppd = mean(lppd),
        bde_lppd_sd = sd(lppd),
        bde_rmse = mean(rmse),
        bde_rmse_sd = sd(rmse),
        bde_cal_error = mean(cal_error),
        bde_cal_error_sd = sd(cal_error),
        de_lppd_sd = sd(de_lppd),
        de_lppd = mean(de_lppd),
        de_rmse_sd = sd(de_rmse),
        de_rmse = mean(de_rmse),
        de_cal_error_sd = sd(de_cal_error),
        de_cal_error = mean(de_cal_error),
        de_time_sd = sd(de_time/ 60),
        de_time = mean(de_time/ 60),
        sampling_time_sd = sd(sampling_time / 60),
        sampling_time = mean(sampling_time / 60),
        # step_size = mean(step_size),
        # L = mean(L),
    ) |>
    arrange(data) |>
    View()

results |>
    group_by(data) |>
    summarize(
        bde_lppd = mean(lppd),
        bde_lppd_sd = sd(lppd),
        bde_rmse = mean(rmse),
        bde_rmse_sd = sd(rmse),
        bde_cal_error = mean(cal_error),
        bde_cal_error_sd = sd(cal_error),
        de_lppd_sd = sd(de_lppd),
        de_lppd = mean(de_lppd),
        de_rmse_sd = sd(de_rmse),
        de_rmse = mean(de_rmse),
        de_cal_error_sd = sd(de_cal_error),
        de_cal_error = mean(de_cal_error),
        de_time_sd = sd(de_time/ 60),
        de_time = mean(de_time/ 60),
        sampling_time_sd = sd(sampling_time / 60),
        sampling_time = mean(sampling_time / 60)
    ) |>
    arrange(data) |>
    pull(de_cal_error) |>
    round(3)

# UCI Grad eval robustness #############################################################
DIR = "results/repl_uci"
FILTER = "_nuts"

results = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))

n_samples <- results$n_samples[1]
n_warmup <- results$warmup_steps[1]
mclmc_grad_evals_per_chain <- 60000 * 2

nuts_grad_evals_per_chain <- results |>
    mutate(grad_evals_per_chain = avg_integration_steps * (n_samples + n_warmup)) |>
    group_by(data) |>
    summarize(
        mean = mean(grad_evals_per_chain),
        sd = sd(grad_evals_per_chain),
        max = max(grad_evals_per_chain),
    )

grad_eval_plot <- nuts_grad_evals_per_chain |>
    mutate(data = factor(data, levels = rev(unique(data)))) |>
    ggplot(aes(x = mean, y = data)) +
    geom_col(aes(color = "BDE"), fill = NA, size = 0.8) +
    geom_errorbarh(
        aes(xmin = mean - sd, xmax = mean + sd),
        height = 0.4,
        size = 0.8,
        color = "#3D348B"
    ) +
    geom_vline(
        aes(color = "MILE",
        xintercept = mclmc_grad_evals_per_chain),
        linetype = "solid",
        linewidth = 1.2
    ) +
    scale_color_manual(values = c("BDE" = "#3D348B", "MILE" = "#F35B04")) +
    labs(x = "Gradient Evaluations per Chain for 1k Posterior Samples", y = "", color = "") +
    theme(legend.position = "bottom")

grad_eval_plot

ggsave(paste0(DIR, "/grad_evals", FILTER, ".pdf"), grad_eval_plot, width = 8, height = 5)


# Diagnostic Plots #####################################################################
DIR = "results/diagnostics"
FILTER = "_mclmc"
results_mclmc = read.csv(paste0(DIR, "/aggr_diagnostics", FILTER, ".csv"))
FILTER = "_nuts"
results_nuts = read.csv(paste0(DIR, "/aggr_diagnostics", FILTER, ".csv"))

results <- results_mclmc |>
    bind_rows(results_nuts) |>
    rename(layer = "Unnamed..0") |>
    mutate(
        layer_type = stringr::str_split(layer, "\\.", simplify = TRUE)[, 3],
        layer_num = stringr::str_split(layer, "\\.", simplify = TRUE)[, 2],
        layer_num = as.numeric(stringr::str_extract(layer_num, "\\d+")),
    )


diag_mapping <- c(
    "ess" = "Effective Sample Size",
    "bcv" = "Between Chain Variance",
    "wcv" = "Within Chain Variance",
    "crhat" = expression("chainwise " * hat(R))
)
for (diagn in c("ess", "bcv", "wcv", "crhat")) {
    plt <- results |>
        mutate(layer_num = layer_num + 1) |>
        mutate(layer_type = case_when(
            layer_type == "bias" ~ "biases",
            layer_type == "kernel" ~ "weights"
        )) |>
        mutate(sampler = case_when(
            sampler == "MCLMC" ~ "MILE",
            sampler == "NUTS" ~ "BDE"
        )) |>
        select(all_of(c("layer_num", diagn, "layer_type", "sampler", "data", "rng"))) |>
        group_by(layer_num, layer_type, sampler, data) |>
        summarize(mean = mean(!!sym(diagn)), sd = sd(!!sym(diagn))) |>
        ggplot(aes(x = layer_num, y = mean, color = sampler, linetype = layer_type)) +
        geom_line() +
        geom_point(size = 2) +
        geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2) +
        labs(
            x = "Hidden Layer", y = diag_mapping[[diagn]],
            color = "", fill = "", linetype = ""
        ) +
        scale_x_continuous(breaks = seq(1, 7, 1)) +
        scale_linetype_manual(values = c("biases" = "dashed", "weights" = "solid")) +
        scale_color_manual(values = c("MILE" = "#F35B04", "BDE" = "#3D348B")) +
        facet_wrap(~data, scales = "free_y", ncol = 1) +
        theme(legend.position = "bottom")

    ggsave(paste0(DIR, "/", diagn, FILTER, ".pdf"), plt, width = 6, height = 6)
}


# Coverage Plots #######################################################################
DIR = "results/repl_uci"
FILTER = "_mclmc"
results_mclmc = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
FILTER = "_nuts"
results_nuts = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))
FILTER = "_de"
results_de = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))

coverage_levels <- colnames(results_de) |>
    stringr::str_subset("de_coverage_\\d+\\.\\d+") |>
    stringr::str_extract("\\d+\\.\\d+") |>
    as.numeric()

coverage_df <- results_mclmc |> select(data, rng, starts_with("coverage_")) |>
    pivot_longer(cols = starts_with("coverage_"), names_to = "nominal_coverage", values_to = "empirical_coverage") |>
    mutate(sampler = "MILE") |>
    bind_rows(
        results_nuts |> select(data, rng, starts_with("coverage_")) |>
            pivot_longer(cols = starts_with("coverage_"), names_to = "nominal_coverage", values_to = "empirical_coverage") |>
            mutate(sampler = "BDE")
    ) |>
    bind_rows(
        results_de |> select(data, rng, starts_with("de_coverage_")) |>
            pivot_longer(cols = starts_with("de_coverage_"), names_to = "nominal_coverage", values_to = "empirical_coverage") |>
            mutate(sampler = "DE")
    ) |>
    mutate(
        nominal_coverage = as.numeric(stringr::str_extract(nominal_coverage, "\\d+\\.\\d+")),
        empirical_coverage = as.numeric(empirical_coverage)
    )

coverage_df |>
    group_by(data, sampler, nominal_coverage) |>
    summarize(mean = mean(empirical_coverage), sd = sd(empirical_coverage)) |>
    arrange(nominal_coverage) |>
    View()

coverage_plot <- coverage_df |>
    group_by(sampler, nominal_coverage) |>
    summarize(
        mean = mean(empirical_coverage),
        sd = sd(empirical_coverage)
    ) |>
    ggplot(aes(x = nominal_coverage, y = mean, color = sampler)) +
    geom_line(linewidth = 1) +
    geom_ribbon(
        aes(ymin = mean - sd, ymax = mean + sd, fill = sampler),
        alpha = 0.2,
        color = NA
    ) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(x = "Nominal Coverage", y = "Empirical Coverage", color = "", fill = "") +
    scale_color_manual(values = c("MILE" = "#F35B04", "BDE" = "#3D348B", "DE" = "grey")) +
    scale_fill_manual(values = c("MILE" = "#F35B04", "BDE" = "#3D348B", "DE" = "grey")) +
    scale_x_continuous(breaks = coverage_levels) +
    scale_y_continuous(breaks = coverage_levels) +
    theme(aspect.ratio = 1)
coverage_plot

data_coverage_plot <- coverage_df |>
    group_by(data, sampler, nominal_coverage) |>
    summarize(
        mean = mean(empirical_coverage),
        sd = sd(empirical_coverage)
    ) |>
    mutate(sampler = factor(sampler, levels = c("DE", "BDE", "MILE"))) |>
    ggplot(aes(x = nominal_coverage, y = mean, color = sampler)) +
    geom_line(linewidth = 1) +
    geom_ribbon(
        aes(ymin = mean - sd, ymax = mean + sd, fill = sampler),
        alpha = 0.2,
        color = NA
    ) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(x = "Nominal Coverage", y = "Empirical Coverage", color = "", fill = "") +
    scale_color_manual(values = c("MILE" = "#F35B04", "BDE" = "#3D348B", "DE" = "grey")) +
    scale_fill_manual(values = c("MILE" = "#F35B04", "BDE" = "#3D348B", "DE" = "grey")) +
    scale_x_continuous(breaks = coverage_levels) +
    scale_y_continuous(breaks = coverage_levels) +
    facet_wrap(~data, ncol = 3) +
    theme(
        aspect.ratio = 1,
        axis.text.x = element_text(angle = 50, hjust = 1, size = fontsize_ticks_legends),
        panel.grid.minor = element_blank()
    )

data_coverage_plot

ggsave(paste0(DIR, "/coverage", ".pdf"), data_coverage_plot, width = 12, height = 7)

# UCI tabular classid #############################################################
DIR = "results/tabular_classif"
FILTER = "_uci"

results = read.csv(paste0(DIR, "/aggr_results", FILTER, ".csv"))

results |>
    group_by(data) |>
    summarize(
        mean_acc = mean(acc), sd_acc = sd(acc),
        mean_lppd = mean(lppd), sd_lppd = sd(lppd),
        mean_nll = mean(nll), sd_nll = sd(nll),
        mean_time = mean(total_time / 60), sd_time = sd(total_time / 60)
    )