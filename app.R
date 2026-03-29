library(shiny)
library(ggplot2)
library(plotly) 

# ==========================================
# 1. DUMMY DATA GENERATORS (The "Placeholders")
# Your team will replace these functions later
# ==========================================

#Simulates the 30-sec RV bins
get_rv_data <- function(stock_id) {
  #In reality, Gia will output a CSV that you load here:
  #df <- read.csv(paste0("data/", stock_id, "_rv.csv"))
  data.frame(
    time = seq(as.POSIXct("2026-03-30 10:00:00"), by = "30 sec", length.out = 100),
    rv = abs(rnorm(100, mean = 0.005, sd = 0.001))
  )
}

# Simulates model evaluation metrics (MSE, QLIKE)
get_evaluation_metrics <- function() {
  data.frame(
    Model = rep(c("HAR-RV", "WLS", "ARMA-GARCH"), each = 20),
    QLIKE = c(abs(rnorm(20, 0.2, 0.05)), abs(rnorm(20, 0.25, 0.08)), abs(rnorm(20, 0.15, 0.04)))
  )
}

# ==========================================
# 2. USER INTERFACE (The Skeleton)
# ==========================================
ui <- navbarPage("Market Maker Volatility Predictor",
                 
                 # Tab 1: Educational / Data Processing Pipeline
                 tabPanel("1. The Order Book & RV",
                          sidebarLayout(
                            sidebarPanel(
                              selectInput("stock_select", "Select Asset:", choices = paste0("stock_", 0:10)),
                              helpText("This tab justifies our data processing pipeline. We convert raw tick data to Weighted Average Price (WAP), then compute the Realised Volatility.")
                            ),
                            mainPanel(
                              h3("Realised Volatility Trajectory"),
                              plotlyOutput("rv_plot")
                            )
                          )
                 ),
                 
                 # Tab 2: Model Justification (Boxplots)
                 tabPanel("2. Model Justification",
                          fluidPage(
                            h3("Validation Set Performance"),
                            p("A market maker must heavily penalize under-predicted volatility. QLIKE is our primary metric."),
                            plotOutput("boxplot_metrics")
                          )
                 ),
                 
                 # Tab 3: The Trading Tool (Forecasts)
                 tabPanel("3. The Trading Desk",
                          sidebarLayout(
                            sidebarPanel(
                              selectInput("model_select", "Active Model:", choices = c("HAR-RV", "WLS", "ARMA-GARCH")),
                              actionButton("predict_btn", "Generate Next-Step Forecast", class = "btn-primary")
                            ),
                            mainPanel(
                              h3("Live Forecast vs Actual"),
                              plotOutput("forecast_plot"),
                              hr(),
                              h4("Execution Metrics"),
                              verbatimTextOutput("current_metrics")
                            )
                          )
                 ),
                 
                 # Tab 4: Insights & Clustering
                 tabPanel("4. Market Insights",
                          fluidPage(
                            h3("Asset Clustering by Volatility Profile"),
                            p("Identifying pairs for hedging strategies."),
                            plotOutput("cluster_plot")
                          )
                 )
)

# ==========================================
# 3. SERVER LOGIC (The Engine)
# ==========================================
server <- function(input, output, session) {
  
  # Reactive block: Re-runs only when the user changes the stock dropdown
  current_stock_data <- reactive({
    get_rv_data(input$stock_select)
  })
  
  # Render Tab 1 Plot (Interactive Plotly)
  output$rv_plot <- renderPlotly({
    df <- current_stock_data()
    p <- ggplot(df, aes(x = time, y = rv)) +
      geom_line(color = "steelblue", size = 1) +
      theme_minimal() +
      labs(x = "Time", y = "Realised Volatility")
    ggplotly(p)
  })
  
  # Render Tab 2 Plot (Boxplots)
  output$boxplot_metrics <- renderPlot({
    eval_data <- get_evaluation_metrics()
    ggplot(eval_data, aes(x = Model, y = QLIKE, fill = Model)) +
      geom_boxplot(alpha = 0.7) +
      theme_minimal() +
      labs(title = "QLIKE Loss Distribution (Lower is Better)")
  })
  
  # Render Tab 3 Plot (Master Forecast)
  output$forecast_plot <- renderPlot({
    # Require the user to click the button to update this specific plot
    input$predict_btn 
    
    # isolate() prevents the plot from updating UNLESS the button is clicked
    isolate({
      df <- current_stock_data()
      # Simulating a train/test split visual (80% train, 20% forecast)
      split_point <- df$time[80] 
      
      ggplot(df, aes(x = time, y = rv)) +
        geom_line(color = "black") +
        geom_vline(xintercept = as.numeric(split_point), linetype = "dashed", color = "red", size = 1) +
        annotate("text", x = split_point, y = max(df$rv), label = "Forecast Window ->", color = "red", hjust = -0.1) +
        theme_minimal()
    })
  })
  
  #Render Tab 3 Text Metrics
  output$current_metrics <- renderText({
    input$predict_btn
    isolate({
      df <- current_stock_data()
      latest_rv <- round(tail(df$rv, 1), 6)
      paste("Active Asset:", input$stock_select, 
            "\nAlgorithm:", input$model_select,
            "\nTarget Volatility Output:", latest_rv)
    })
  })
  
  #Render Tab 4 Plot (Clustering)
  output$cluster_plot <- renderPlot({
    cluster_data <- data.frame(PC1 = rnorm(50), PC2 = rnorm(50), Cluster = as.factor(sample(1:3, 50, replace = TRUE)))
    ggplot(cluster_data, aes(x = PC1, y = PC2, color = Cluster)) +
      geom_point(size = 4, alpha = 0.8) +
      theme_minimal() 
  })
}


shinyApp(ui = ui, server = server)