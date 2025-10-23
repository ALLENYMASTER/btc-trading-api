import SwiftUI

// ============================================================================
// MARK: - API Client
// ============================================================================

class BTCTradingAPI: ObservableObject {
    // Change this to your server URL
    // For local testing: "http://localhost:8000"
    // For cloud: "https://your-server.com"
    private let baseURL = "https://web-production-dd9d.up.railway.app/"
    
    @Published var currentPrice: Double?
    @Published var priceChange24h: Double?
    @Published var signal: TradingSignal?
    @Published var news: [NewsItem] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    // MARK: - Data Models
    
    struct PriceData: Codable {
        let price: Double
        let change_24h: Double
        let timestamp: String
    }
    
    struct TradingSignal: Codable, Identifiable {
        var id = UUID()
        let prediction: String
        let confidence: Double
        let prob_up: Double
        let prob_down: Double
        let current_price: Double
        let sentiment: Double
        let events: Int
        let timestamp: String
        let recommendation: String
        
        enum CodingKeys: String, CodingKey {
            case prediction, confidence, prob_up, prob_down
            case current_price, sentiment, events, timestamp, recommendation
        }
    }
    
    struct NewsItem: Codable, Identifiable {
        var id = UUID()
        let title: String
        let sentiment: Double
        let published_at: String
        let source: String
        let events: [String]
        
        enum CodingKeys: String, CodingKey {
            case title, sentiment, published_at, source, events
        }
    }
    
    struct BacktestResult: Codable {
        let total_return: Double
        let win_rate: Double
        let total_trades: Int
        let final_capital: Double
        let sl_hits: Int
        let tp_hits: Int
    }
    
    // MARK: - API Methods
    
    func fetchPrice() async {
        await performRequest(endpoint: "/price", type: PriceData.self) { data in
            DispatchQueue.main.async {
                self.currentPrice = data.price
                self.priceChange24h = data.change_24h
            }
        }
    }
    
    func fetchSignal() async {
        await performRequest(endpoint: "/signal", type: TradingSignal.self) { signal in
            DispatchQueue.main.async {
                self.signal = signal
            }
        }
    }
    
    func fetchNews() async {
        await performRequest(endpoint: "/news?limit=10", type: [NewsItem].self) { news in
            DispatchQueue.main.async {
                self.news = news
            }
        }
    }
    
    func trainModel() async {
        guard let url = URL(string: "\(baseURL)/model/train") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                DispatchQueue.main.async {
                    self.errorMessage = "Model training started"
                }
            }
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "Training failed: \(error.localizedDescription)"
            }
        }
    }
    
    private func performRequest<T: Decodable>(
        endpoint: String,
        type: T.Type,
        completion: @escaping (T) -> Void
    ) async {
        guard let url = URL(string: "\(baseURL)\(endpoint)") else {
            DispatchQueue.main.async {
                self.errorMessage = "Invalid URL"
            }
            return
        }
        
        DispatchQueue.main.async {
            self.isLoading = true
            self.errorMessage = nil
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw URLError(.badServerResponse)
            }
            
            let decoded = try JSONDecoder().decode(T.self, from: data)
            completion(decoded)
            
            DispatchQueue.main.async {
                self.isLoading = false
            }
        } catch {
            DispatchQueue.main.async {
                self.isLoading = false
                self.errorMessage = error.localizedDescription
            }
        }
    }
}

// ============================================================================
// MARK: - Main App View
// ============================================================================

struct BTCTradingAppView: View {
    @StateObject private var api = BTCTradingAPI()
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView(api: api)
                .tabItem {
                    Label("Dashboard", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(0)
            
            SignalView(api: api)
                .tabItem {
                    Label("Signal", systemImage: "scope")
                }
                .tag(1)
            
            NewsView(api: api)
                .tabItem {
                    Label("News", systemImage: "newspaper")
                }
                .tag(2)
            
            SettingsView(api: api)
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(3)
        }
        .accentColor(.orange)
    }
}

// ============================================================================
// MARK: - Dashboard View
// ============================================================================

struct DashboardView: View {
    @ObservedObject var api: BTCTradingAPI
    @State private var autoRefresh = true
    let timer = Timer.publish(every: 60, on: .main, in: .common).autoconnect()
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Price Card
                    PriceCard(
                        price: api.currentPrice,
                        change: api.priceChange24h
                    )
                    
                    // Signal Summary Card
                    if let signal = api.signal {
                        SignalSummaryCard(signal: signal)
                    }
                    
                    // Quick Actions
                    QuickActionsCard(api: api)
                    
                    // Error Message
                    if let error = api.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding()
                    }
                }
                .padding()
            }
            .navigationTitle("BTC Trading")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { Task { await refreshAll() } }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .onReceive(timer) { _ in
                if autoRefresh {
                    Task { await refreshAll() }
                }
            }
        }
        .task {
            await refreshAll()
        }
    }
    
    func refreshAll() async {
        await api.fetchPrice()
        await api.fetchSignal()
    }
}

// ============================================================================
// MARK: - Price Card
// ============================================================================

struct PriceCard: View {
    let price: Double?
    let change: Double?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Bitcoin")
                .font(.headline)
                .foregroundColor(.secondary)
            
            if let price = price {
                HStack(alignment: .firstTextBaseline) {
                    Text("$\(price, specifier: "%.2f")")
                        .font(.system(size: 36, weight: .bold))
                    
                    Spacer()
                    
                    if let change = change {
                        HStack(spacing: 4) {
                            Image(systemName: change >= 0 ? "arrow.up" : "arrow.down")
                            Text("\(abs(change), specifier: "%.2f")%")
                        }
                        .foregroundColor(change >= 0 ? .green : .red)
                        .font(.title3)
                    }
                }
            } else {
                ProgressView()
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
}

// ============================================================================
// MARK: - Signal Summary Card
// ============================================================================

struct SignalSummaryCard: View {
    let signal: BTCTradingAPI.TradingSignal
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Text("Trading Signal")
                    .font(.headline)
                
                Spacer()
                
                Text(signal.recommendation)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(recommendationColor)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(recommendationColor.opacity(0.2))
                    .cornerRadius(8)
            }
            
            Divider()
            
            HStack {
                VStack(alignment: .leading) {
                    Text("PREDICTION")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(signal.prediction)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(signal.prediction == "UP" ? .green : .red)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("CONFIDENCE")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(Int(signal.confidence * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                }
            }
            
            // Probability Bar
            HStack(spacing: 5) {
                GeometryReader { geometry in
                    HStack(spacing: 0) {
                        Rectangle()
                            .fill(Color.red)
                            .frame(width: geometry.size.width * CGFloat(signal.prob_down))
                        
                        Rectangle()
                            .fill(Color.green)
                            .frame(width: geometry.size.width * CGFloat(signal.prob_up))
                    }
                }
                .frame(height: 20)
                .cornerRadius(10)
            }
            
            HStack {
                Text("↓ \(Int(signal.prob_down * 100))%")
                    .font(.caption)
                    .foregroundColor(.red)
                Spacer()
                Text("↑ \(Int(signal.prob_up * 100))%")
                    .font(.caption)
                    .foregroundColor(.green)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
    
    var recommendationColor: Color {
        if signal.recommendation.contains("STRONG") {
            return .orange
        } else if signal.recommendation.contains("MODERATE") {
            return .blue
        } else {
            return .gray
        }
    }
}

// ============================================================================
// MARK: - Quick Actions Card
// ============================================================================

struct QuickActionsCard: View {
    @ObservedObject var api: BTCTradingAPI
    
    var body: some View {
        VStack(spacing: 15) {
            Text("Quick Actions")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            HStack(spacing: 15) {
                ActionButton(
                    title: "Refresh",
                    icon: "arrow.clockwise",
                    color: .blue
                ) {
                    Task {
                        await api.fetchPrice()
                        await api.fetchSignal()
                    }
                }
                
                ActionButton(
                    title: "News",
                    icon: "newspaper",
                    color: .orange
                ) {
                    Task {
                        await api.fetchNews()
                    }
                }
                
                ActionButton(
                    title: "Train",
                    icon: "brain",
                    color: .purple
                ) {
                    Task {
                        await api.trainModel()
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
}

struct ActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.title2)
                Text(title)
                    .font(.caption)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .background(color.opacity(0.1))
            .foregroundColor(color)
            .cornerRadius(10)
        }
    }
}

// ============================================================================
// MARK: - Signal View
// ============================================================================

struct SignalView: View {
    @ObservedObject var api: BTCTradingAPI
    
    var body: some View {
        NavigationView {
            ScrollView {
                if let signal = api.signal {
                    VStack(spacing: 20) {
                        DetailedSignalCard(signal: signal)
                        
                        // Sentiment Analysis
                        SentimentCard(
                            sentiment: signal.sentiment,
                            events: signal.events
                        )
                    }
                    .padding()
                } else if api.isLoading {
                    ProgressView()
                } else {
                    Text("No signal available")
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Trading Signal")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { Task { await api.fetchSignal() } }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
        .task {
            await api.fetchSignal()
        }
    }
}

struct DetailedSignalCard: View {
    let signal: BTCTradingAPI.TradingSignal
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Detailed Analysis")
                .font(.title2)
                .fontWeight(.bold)
            
            VStack(spacing: 15) {
                DetailRow(label: "Prediction", value: signal.prediction)
                DetailRow(label: "Confidence", value: "\(Int(signal.confidence * 100))%")
                DetailRow(label: "Current Price", value: "$\(signal.current_price, specifier: "%.2f")")
                DetailRow(label: "Recommendation", value: signal.recommendation)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
}

struct DetailRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.semibold)
        }
    }
}

struct SentimentCard: View {
    let sentiment: Double
    let events: Int
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Market Sentiment")
                .font(.headline)
            
            HStack {
                VStack(alignment: .leading) {
                    Text("SENTIMENT SCORE")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.3f", sentiment))
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(sentimentColor)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("MARKET EVENTS")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(events)")
                        .font(.title)
                        .fontWeight(.bold)
                }
            }
            
            // Sentiment bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 10)
                        .cornerRadius(5)
                    
                    Rectangle()
                        .fill(sentimentColor)
                        .frame(width: geometry.size.width * CGFloat((sentiment + 1) / 2), height: 10)
                        .cornerRadius(5)
                }
            }
            .frame(height: 10)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
    
    var sentimentColor: Color {
        if sentiment > 0.2 { return .green }
        else if sentiment < -0.2 { return .red }
        else { return .orange }
    }
}

// ============================================================================
// MARK: - News View
// ============================================================================

struct NewsView: View {
    @ObservedObject var api: BTCTradingAPI
    
    var body: some View {
        NavigationView {
            List(api.news) { article in
                NewsRowView(article: article)
            }
            .listStyle(InsetGroupedListStyle())
            .navigationTitle("Latest News")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { Task { await api.fetchNews() } }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
        .task {
            await api.fetchNews()
        }
    }
}

struct NewsRowView: View {
    let article: BTCTradingAPI.NewsItem
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(article.title)
                .font(.headline)
                .lineLimit(2)
            
            HStack {
                // Sentiment indicator
                HStack(spacing: 4) {
                    Circle()
                        .fill(sentimentColor)
                        .frame(width: 8, height: 8)
                    Text(sentimentText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                // Events badges
                ForEach(article.events, id: \.self) { event in
                    Text(event)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(eventColor(event).opacity(0.2))
                        .foregroundColor(eventColor(event))
                        .cornerRadius(4)
                }
            }
            
            HStack {
                Text(article.source)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(formatDate(article.published_at))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
    
    var sentimentColor: Color {
        if article.sentiment > 0.2 { return .green }
        else if article.sentiment < -0.2 { return .red }
        else { return .orange }
    }
    
    var sentimentText: String {
        if article.sentiment > 0.2 { return "Bullish" }
        else if article.sentiment < -0.2 { return "Bearish" }
        else { return "Neutral" }
    }
    
    func eventColor(_ event: String) -> Color {
        switch event {
        case "Regulation": return .red
        case "Institutional": return .blue
        case "Security": return .purple
        default: return .gray
        }
    }
    
    func formatDate(_ isoString: String) -> String {
        let formatter = ISO8601DateFormatter()
        guard let date = formatter.date(from: isoString) else { return "" }
        
        let displayFormatter = DateFormatter()
        displayFormatter.dateFormat = "MMM d, HH:mm"
        return displayFormatter.string(from: date)
    }
}

// ============================================================================
// MARK: - Settings View
// ============================================================================

struct SettingsView: View {
    @ObservedObject var api: BTCTradingAPI
    @State private var showingBacktest = false
    @State private var backtestDays = 30
    @State private var backtestResult: BTCTradingAPI.BacktestResult?
    @State private var isBacktesting = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model")) {
                    Button(action: {
                        Task { await api.trainModel() }
                    }) {
                        HStack {
                            Image(systemName: "brain")
                            Text("Retrain Model")
                        }
                    }
                    
                    Button(action: { showingBacktest = true }) {
                        HStack {
                            Image(systemName: "chart.bar")
                            Text("Run Backtest")
                        }
                    }
                }
                
                Section(header: Text("Refresh")) {
                    Button(action: {
                        Task {
                            await api.fetchPrice()
                            await api.fetchSignal()
                            await api.fetchNews()
                        }
                    }) {
                        HStack {
                            Image(systemName: "arrow.clockwise")
                            Text("Refresh All Data")
                        }
                    }
                }
                
                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Model")
                        Spacer()
                        Text("Random Forest")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .sheet(isPresented: $showingBacktest) {
                BacktestView(
                    days: $backtestDays,
                    result: $backtestResult,
                    isBacktesting: $isBacktesting,
                    api: api
                )
            }
        }
    }
}

// ============================================================================
// MARK: - Backtest View
// ============================================================================

struct BacktestView: View {
    @Environment(\.dismiss) var dismiss
    @Binding var days: Int
    @Binding var result: BTCTradingAPI.BacktestResult?
    @Binding var isBacktesting: Bool
    @ObservedObject var api: BTCTradingAPI
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                if isBacktesting {
                    ProgressView("Running backtest...")
                        .padding()
                } else if let result = result {
                    ScrollView {
                        VStack(spacing: 20) {
                            BacktestResultCard(result: result)
                        }
                        .padding()
                    }
                } else {
                    VStack(spacing: 20) {
                        Text("Backtest Configuration")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        VStack(alignment: .leading) {
                            Text("Days: \(days)")
                                .font(.headline)
                            Slider(value: Binding(
                                get: { Double(days) },
                                set: { days = Int($0) }
                            ), in: 7...90, step: 1)
                        }
                        .padding()
                        
                        Button(action: runBacktest) {
                            Text("Run Backtest")
                                .fontWeight(.semibold)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                        .padding(.horizontal)
                    }
                }
                
                Spacer()
            }
            .navigationTitle("Backtest")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
    
    func runBacktest() {
        isBacktesting = true
        result = nil
        
        Task {
            guard let url = URL(string: "\(api.baseURL)/backtest") else { return }
            
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let body = ["days": days, "initial_capital": 10000] as [String : Any]
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)
            
            do {
                let (data, _) = try await URLSession.shared.data(for: request)
                let decoded = try JSONDecoder().decode(BTCTradingAPI.BacktestResult.self, from: data)
                
                DispatchQueue.main.async {
                    self.result = decoded
                    self.isBacktesting = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.isBacktesting = false
                    self.api.errorMessage = "Backtest failed: \(error.localizedDescription)"
                }
            }
        }
    }
}

struct BacktestResultCard: View {
    let result: BTCTradingAPI.BacktestResult
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Backtest Results")
                .font(.title2)
                .fontWeight(.bold)
            
            // Return Card
            VStack(spacing: 10) {
                Text("Total Return")
                    .font(.headline)
                    .foregroundColor(.secondary)
                
                Text("\(result.total_return, specifier: "%.2f")%")
                    .font(.system(size: 48, weight: .bold))
                    .foregroundColor(result.total_return >= 0 ? .green : .red)
                
                Text("$10,000 → $\(result.final_capital, specifier: "%.2f")")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(15)
            .shadow(radius: 5)
            
            // Stats Grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 15) {
                StatBox(label: "Win Rate", value: "\(Int(result.win_rate * 100))%")
                StatBox(label: "Total Trades", value: "\(result.total_trades)")
                StatBox(label: "Stop Loss", value: "\(result.sl_hits)")
                StatBox(label: "Take Profit", value: "\(result.tp_hits)")
            }
        }
    }
}

struct StatBox: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack(spacing: 8) {
            Text(value)
                .font(.title2)
                .fontWeight(.bold)
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 3)
    }
}

// ============================================================================
// MARK: - Preview
// ============================================================================

struct BTCTradingAppView_Previews: PreviewProvider {
    static var previews: some View {
        BTCTradingAppView()
    }
}
