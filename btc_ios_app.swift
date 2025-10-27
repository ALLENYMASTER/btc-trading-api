import SwiftUI
import Foundation
import Combine
import Charts

// ============================================================================
// MARK: - API Client (原有代碼，保持不變)
// ============================================================================
class BTCTradingAPI: ObservableObject {
    private let baseURL = "https://web-production-dd9d.up.railway.app"

    @Published var currentPrice: Double?
    @Published var priceChange24h: Double?
    @Published var signal: TradingSignal?
    @Published var news: [NewsItem] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var apiStatus: String = "Checking..."

    // MARK: - Data Models
    struct HealthResponse: Codable {
        let status: String
        let model_ready: Bool?
        let model_training: Bool?
    }
    
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
    
    // Signal History
    private let historyKey = "signalHistory"
    private let maxHistoryCount = 100

    @Published var signalHistory: [TradingSignal] = [] {
        didSet {
            saveSignalHistory()
        }
    }

    init() {
        loadSignalHistory()
    }

    private func saveSignalHistory() {
        let encoder = JSONEncoder()
        let recordsToSave = Array(signalHistory.prefix(maxHistoryCount))
        
        do {
            let data = try encoder.encode(recordsToSave)
            UserDefaults.standard.set(data, forKey: historyKey)
            UserDefaults.standard.synchronize()
        } catch {
            print("❌ Failed to save signal history: \(error)")
        }
    }

    func loadSignalHistory() {
        guard let data = UserDefaults.standard.data(forKey: historyKey) else { return }
        let decoder = JSONDecoder()
        do {
            let history = try decoder.decode([TradingSignal].self, from: data)
            DispatchQueue.main.async {
                self.signalHistory = history
            }
        } catch {
            print("❌ Failed to load signal history: \(error)")
        }
    }

    func forceSaveHistory() {
        saveSignalHistory()
    }

    // API Methods
    func checkConnection() async {
        await performRequest(endpoint: "/", type: HealthResponse.self) { response in
            DispatchQueue.main.async {
                self.apiStatus = response.status == "online" ? "Online ✓" : "Offline ✗"
            }
        }
    }
    
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

                let isDuplicate = self.signalHistory.first?.timestamp == signal.timestamp
                
                if !isDuplicate {
                    self.signalHistory.insert(signal, at: 0)
                    if self.signalHistory.count > self.maxHistoryCount {
                        self.signalHistory = Array(self.signalHistory.prefix(self.maxHistoryCount))
                    }
                }

                if signal.prediction == "WAIT" {
                    self.errorMessage = signal.recommendation
                }
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
            var request = URLRequest(url: url)
            request.timeoutInterval = 30
            request.httpMethod = "GET"
            request.setValue("application/json", forHTTPHeaderField: "Accept")
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw URLError(.badServerResponse)
            }
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw URLError(.badServerResponse)
            }
            
            let decoded = try JSONDecoder().decode(T.self, from: data)
            completion(decoded)
            
            DispatchQueue.main.async {
                self.isLoading = false
                self.apiStatus = "Connected ✓"
            }
            
        } catch {
            DispatchQueue.main.async {
                self.isLoading = false
                self.errorMessage = error.localizedDescription
                self.apiStatus = "Error ✗"
            }
        }
    }
}

// ============================================================================
// MARK: - 🆕 位置 1: TradingSimulator (Edge AI Engine)
// ============================================================================

// User Profile
struct UserProfile: Codable {
    var tradingStyle: TradingStyle = .balanced
    var riskTolerance: Double = 0.5
    var avgHoldingTime: TimeInterval = 3600
    var preferredTimeOfDay: Int = 12
    var emotionalFactors: EmotionalFactors = EmotionalFactors()
    var optimalHoldingTime: TimeInterval = 3600
    var maxProfitableHoldTime: TimeInterval = 14400
    
    enum TradingStyle: String, Codable, CaseIterable {
        case aggressive = "Aggressive"
        case balanced = "Balanced"
        case conservative = "Conservative"
    }
    
    struct EmotionalFactors: Codable {
        var panicSellThreshold: Double = -5.0
        var greedHoldThreshold: Double = 10.0
        var fomo: Double = 0.5
        var patience: Double = 0.5
    }
}

// Trade Model
struct Trade: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let type: TradeType
    let entryPrice: Double
    let exitPrice: Double?
    let quantity: Double
    let pnl: Double?
    let pnlPercent: Double?
    let signal: SignalInfo
    
    enum TradeType: String, Codable {
        case buy = "BUY"
        case sell = "SELL"
    }
    
    struct SignalInfo: Codable {
        let prediction: String
        let confidence: Double
        let sentiment: Double
    }
    
    init(id: UUID = UUID(), timestamp: Date = Date(), type: TradeType,
         entryPrice: Double, exitPrice: Double? = nil, quantity: Double,
         pnl: Double? = nil, pnlPercent: Double? = nil, signal: SignalInfo) {
        self.id = id
        self.timestamp = timestamp
        self.type = type
        self.entryPrice = entryPrice
        self.exitPrice = exitPrice
        self.quantity = quantity
        self.pnl = pnl
        self.pnlPercent = pnlPercent
        self.signal = signal
    }
}

struct PerformanceSnapshot: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let equity: Double
    let trades: Int
    let winRate: Double
    
    init(id: UUID = UUID(), timestamp: Date = Date(),
         equity: Double, trades: Int, winRate: Double) {
        self.id = id
        self.timestamp = timestamp
        self.equity = equity
        self.trades = trades
        self.winRate = winRate
    }
}

// TradingSimulator
class TradingSimulator: ObservableObject {
    static let shared = TradingSimulator(initialCapital: 10000)
    
    private var cash: Double
    @Published var initialCapital: Double
    @Published var trades: [Trade] = []
    @Published var equityCurve: [PerformanceSnapshot] = []
    @Published var isEnabled: Bool = true
    @Published var userProfile: UserProfile = UserProfile()
    @Published var aiPersonality: String = "🤖 Learning..."
    
    @Published var signalProcessCount: Int = 0
    @Published var rejectedSignalCount: Int = 0
    @Published var lastRejectionReason: String = ""
    @Published var lastProcessedSignal: Date?
    @Published var lastPriceUpdate: Double?
    
    @Published var positions: [Trade] = []
    @Published var lastTradeTime: Date?
    
    let maxPositions: Int = 5
    let positionSizePct: Double = 0.20
    let minTradeInterval: TimeInterval = 1800
    
    private let feeRate: Double = 0.001
    private let storageKey = "TradingSimulator"
    private let profileKey = "UserProfile"
    
    var currentEquity: Double {
        return calculateTotalEquity(currentPrice: lastPriceUpdate)
    }
    
    var canOpenNewPosition: Bool {
        guard positions.count < maxPositions else { return false }
        
        if let lastTime = lastTradeTime {
            let elapsed = Date().timeIntervalSince(lastTime)
            guard elapsed >= minTradeInterval else { return false }
        }
        
        return true
    }
    
    var cooldownRemaining: TimeInterval {
        guard let lastTime = lastTradeTime else { return 0 }
        let elapsed = Date().timeIntervalSince(lastTime)
        return max(0, minTradeInterval - elapsed)
    }

    var cooldownText: String {
        let remaining = cooldownRemaining
        if remaining <= 0 { return "Ready" }
        
        let minutes = Int(remaining / 60)
        let seconds = Int(remaining.truncatingRemainder(dividingBy: 60))
        
        if minutes > 0 {
            return "\(minutes) min \(seconds) sec"
        } else {
            return "\(seconds) sec"
        }
    }
    
    func calculateTotalEquity(currentPrice: Double?) -> Double {
        var equity = cash
        if let price = currentPrice {
            for position in positions {
                equity += position.quantity * price
            }
        }
        return equity
    }
    
    var cumulativeReturn: Double {
        ((currentEquity - initialCapital) / initialCapital) * 100
    }
    
    var winRate: Double {
        let closedTrades = trades.filter { $0.exitPrice != nil }
        guard !closedTrades.isEmpty else { return 0 }
        let wins = closedTrades.filter { ($0.pnl ?? 0) > 0 }.count
        return Double(wins) / Double(closedTrades.count)
    }
    
    var totalTrades: Int { trades.count }
    var totalClosedTrades: Int { trades.filter { $0.exitPrice != nil }.count }
    
    var profitFactor: Double {
        let closedTrades = trades.filter { $0.exitPrice != nil }
        let grossProfit = closedTrades.filter { ($0.pnl ?? 0) > 0 }
            .reduce(0) { $0 + ($1.pnl ?? 0) }
        let grossLoss = abs(closedTrades.filter { ($0.pnl ?? 0) < 0 }
            .reduce(0) { $0 + ($1.pnl ?? 0) })
        guard grossLoss > 0 else { return grossProfit > 0 ? 999 : 0 }
        return grossProfit / grossLoss
    }
    
    var maxDrawdown: Double {
        return getMaxDrawdown(currentPrice: lastPriceUpdate)
    }
    
    func getMaxDrawdown(currentPrice: Double?) -> Double {
        guard !equityCurve.isEmpty else { return 0 }
        
        var allEquities = equityCurve.map { $0.equity }
        let currentEq = calculateTotalEquity(currentPrice: currentPrice)
        allEquities.append(currentEq)
        
        var peak = allEquities[0]
        var maxDD = 0.0
        
        for equity in allEquities {
            if equity > peak { peak = equity }
            let drawdown = ((peak - equity) / peak) * 100
            maxDD = max(maxDD, drawdown)
        }
        
        return maxDD
    }
    
    private init(initialCapital: Double) {
        self.initialCapital = initialCapital
        self.cash = initialCapital
        loadState()
        loadUserProfile()
        updateAIPersonality()
        if equityCurve.isEmpty {
            addSnapshot(currentPrice: nil)
        }
    }
    
    public func captureSnapshot(currentPrice: Double?) {
        addSnapshot(currentPrice: currentPrice)
    }

    public func persistState() {
        saveState()
    }

    public func saveForBackground(currentPrice: Double?) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.addSnapshot(currentPrice: currentPrice)
            self.saveState()
        }
    }

    public func forceRefresh() {
        DispatchQueue.main.async { [weak self] in
            self?.objectWillChange.send()
        }
    }
    
    // 🎯 核心：個人化閾值計算 (Edge AI)
    func calculatePersonalizedThreshold() -> Double {
        var threshold = 0.65
        
        // 1️⃣ 基於風險偏好調整
        threshold -= (userProfile.riskTolerance - 0.5) * 0.2
        
        // 2️⃣ 基於勝率調整
        if winRate > 0.7 {
            threshold -= 0.08
        } else if winRate > 0.6 {
            threshold -= 0.05
        } else if winRate < 0.35 {
            threshold += 0.10
        } else if winRate < 0.4 {
            threshold += 0.05
        }
        
        // 3️⃣ 基於近期表現調整
        let recentTrades = Array(trades.suffix(5).filter { $0.exitPrice != nil })
        if recentTrades.count >= 5 {
            let recentWins = recentTrades.filter { ($0.pnl ?? 0) > 0 }.count
            let recentWinRate = Double(recentWins) / Double(recentTrades.count)
            
            if recentWinRate >= 0.8 {
                threshold -= 0.03
            } else if recentWinRate <= 0.2 {
                threshold += 0.05
            }
        }
        
        let finalThreshold = max(0.50, min(0.80, threshold))
        return finalThreshold
    }
    
    // 處理信號 (Edge AI 決策)
    func processSignal(signal: BTCTradingAPI.TradingSignal, currentPrice: Double) {
        guard isEnabled else { return }
        
        lastPriceUpdate = currentPrice
        signalProcessCount += 1
        lastProcessedSignal = Date()
        
        let adjustedConfidence = adjustConfidenceByStyle(
            baseConfidence: signal.confidence,
            style: userProfile.tradingStyle
        )
        
        let signalInfo = Trade.SignalInfo(
            prediction: signal.prediction,
            confidence: adjustedConfidence,
            sentiment: signal.sentiment
        )
        
        // 檢查平倉
        for (index, position) in positions.enumerated().reversed() {
            if shouldClosePosition(position: position, newSignal: signal, currentPrice: currentPrice) {
                closePosition(at: index, exitPrice: currentPrice)
            }
        }
        
        // 檢查開倉
        if signal.prediction == "UP" {
            let threshold = calculatePersonalizedThreshold()
            
            if adjustedConfidence >= threshold {
                if canOpenNewPosition {
                    openBuyPosition(price: currentPrice, signal: signalInfo, totalEquity: currentEquity)
                } else {
                    rejectedSignalCount += 1
                    
                    if positions.count >= maxPositions {
                        lastRejectionReason = "Max positions reached (\(maxPositions))"
                    } else if let lastTime = lastTradeTime {
                        let elapsed = Date().timeIntervalSince(lastTime)
                        let remaining = Int((minTradeInterval - elapsed) / 60)
                        lastRejectionReason = "Cooldown: \(remaining) min remaining"
                    }
                }
            } else {
                rejectedSignalCount += 1
                lastRejectionReason = "Low confidence"
            }
        }
        
        // 學習機制
        if totalClosedTrades % 10 == 0 && totalClosedTrades > 0 {
            learnFromRecentPerformance()
        }
        
        objectWillChange.send()
        saveState()
        saveUserProfile()
    }
    
    private func adjustConfidenceByStyle(baseConfidence: Double, style: UserProfile.TradingStyle) -> Double {
        switch style {
        case .aggressive: return min(1.0, baseConfidence * 1.2)
        case .conservative: return baseConfidence * 0.8
        case .balanced: return baseConfidence
        }
    }
    
    private func openBuyPosition(price: Double, signal: Trade.SignalInfo, totalEquity: Double) {
        let positionValue = totalEquity * positionSizePct
        let fee = positionValue * feeRate
        let netInvestment = positionValue - fee
        let quantity = netInvestment / price
        
        guard quantity > 0 && cash >= (netInvestment + fee) else { return }
        
        cash -= (netInvestment + fee)
        
        let trade = Trade(type: .buy, entryPrice: price, quantity: quantity, signal: signal)
        
        positions.append(trade)
        trades.append(trade)
        lastTradeTime = Date()
        
        lastPriceUpdate = price
        addSnapshot(currentPrice: price)
        objectWillChange.send()
    }
    
    private func closePosition(at index: Int, exitPrice: Double) {
        guard index < positions.count else { return }
        
        var position = positions[index]
        
        let exitValue = position.quantity * exitPrice
        let fee = exitValue * feeRate
        let netProceeds = exitValue - fee
        
        let entryCost = position.quantity * position.entryPrice
        let pnl = netProceeds - entryCost
        let pnlPercent = (pnl / entryCost) * 100
        
        cash += netProceeds
        
        position = Trade(
            id: position.id, timestamp: position.timestamp, type: position.type,
            entryPrice: position.entryPrice, exitPrice: exitPrice,
            quantity: position.quantity, pnl: pnl, pnlPercent: pnlPercent,
            signal: position.signal
        )
        
        if let tradeIndex = trades.firstIndex(where: { $0.id == position.id }) {
            trades[tradeIndex] = position
        }
        
        positions.remove(at: index)
        lastTradeTime = Date()
        
        lastPriceUpdate = exitPrice
        addSnapshot(currentPrice: exitPrice)
        objectWillChange.send()
    }
    
    private func shouldClosePosition(position: Trade, newSignal: BTCTradingAPI.TradingSignal, currentPrice: Double) -> Bool {
        let unrealizedPnLPct = ((currentPrice - position.entryPrice) / position.entryPrice) * 100
        
        if unrealizedPnLPct <= userProfile.emotionalFactors.panicSellThreshold {
            return true
        }
        
        if position.type == .buy && newSignal.prediction == "DOWN" && newSignal.confidence >= 0.65 {
            return true
        }
        
        return false
    }
    
    // 🧠 自我學習算法
    private func learnFromRecentPerformance() {
        let recentTrades = Array(trades.suffix(10).filter { $0.exitPrice != nil })
        guard !recentTrades.isEmpty else { return }
        
        let recentWinRate = Double(recentTrades.filter { ($0.pnl ?? 0) > 0 }.count) / Double(recentTrades.count)
        
        if recentWinRate > 0.7 {
            userProfile.riskTolerance = min(1.0, userProfile.riskTolerance + 0.05)
        } else if recentWinRate < 0.3 {
            userProfile.riskTolerance = max(0.0, userProfile.riskTolerance - 0.05)
        }
        
        updateAIPersonality()
    }
    
    func updateAIPersonality() {
        let performanceEmoji: String
        let performanceText: String
        
        if totalTrades < 5 {
            performanceEmoji = "🤖"
            performanceText = "Learning"
        } else if winRate > 0.6 {
            performanceEmoji = "⭐"
            performanceText = "Winning"
        } else if winRate < 0.4 {
            performanceEmoji = "📉"
            performanceText = "Adaptive"
        } else {
            performanceEmoji = "📊"
            performanceText = "Developing"
        }
        
        let style = userProfile.tradingStyle.rawValue
        
        let riskText: String
        let risk = userProfile.riskTolerance
        
        if risk > 0.7 {
            riskText = "Risk-Taker"
        } else if risk < 0.3 {
            riskText = "Cautious"
        } else {
            riskText = "Prudent"
        }
        
        let learningStage: String
        if totalTrades < 5 {
            learningStage = "🌱 Beginner"
        } else if totalTrades < 20 {
            learningStage = "📚 Learning"
        } else if totalTrades < 50 {
            learningStage = "🎓 Developing"
        } else {
            learningStage = "🎖️ Expert"
        }
        
        aiPersonality = "\(learningStage) | \(performanceEmoji) \(performanceText) \n\(style) | \(riskText)"
    }
    
    func reset(preservePersonality: Bool = true, setStyle: UserProfile.TradingStyle? = nil) {
        cash = initialCapital
        trades.removeAll()
        equityCurve.removeAll()
        positions.removeAll()
        lastTradeTime = nil
        
        signalProcessCount = 0
        rejectedSignalCount = 0
        lastRejectionReason = ""
        lastPriceUpdate = nil
        
        if !preservePersonality {
            userProfile = UserProfile()
        }
        
        if let style = setStyle {
            userProfile.tradingStyle = style
            
            switch style {
            case .aggressive:
                userProfile.riskTolerance = 0.8
            case .balanced:
                userProfile.riskTolerance = 0.5
            case .conservative:
                userProfile.riskTolerance = 0.3
            }
        }
        
        updateAIPersonality()
        addSnapshot(currentPrice: nil)
        saveState()
        saveUserProfile()
        
        objectWillChange.send()
    }
    
    private func addSnapshot(currentPrice: Double?) {
        let equity = calculateTotalEquity(currentPrice: currentPrice)
        let snapshot = PerformanceSnapshot(
            equity: equity,
            trades: totalTrades,
            winRate: winRate
        )
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.equityCurve.append(snapshot)
            if self.equityCurve.count > 1000 {
                self.equityCurve.removeFirst()
            }
            self.objectWillChange.send()
        }
    }

    private func saveState() {
        let state = SimulatorState(
            cash: cash,
            initialCapital: initialCapital,
            trades: trades,
            equityCurve: equityCurve,
            positions: positions,
            lastTradeTime: lastTradeTime,
            isEnabled: isEnabled,
            lastPriceUpdate: lastPriceUpdate
        )
        if let encoded = try? JSONEncoder().encode(state) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }
    
    private func loadState() {
        guard let data = UserDefaults.standard.data(forKey: storageKey) else { return }
        
        do {
            let state = try JSONDecoder().decode(SimulatorState.self, from: data)
            cash = state.cash
            initialCapital = state.initialCapital
            trades = state.trades
            equityCurve = state.equityCurve
            positions = state.positions
            lastTradeTime = state.lastTradeTime
            isEnabled = state.isEnabled
            lastPriceUpdate = state.lastPriceUpdate
        } catch {
            print("⚠️ Failed to decode state: \(error)")
        }
    }
    
    private func saveUserProfile() {
        if let encoded = try? JSONEncoder().encode(userProfile) {
            UserDefaults.standard.set(encoded, forKey: profileKey)
        }
    }
    
    private func loadUserProfile() {
        guard let data = UserDefaults.standard.data(forKey: profileKey),
              let profile = try? JSONDecoder().decode(UserProfile.self, from: data) else { return }
        userProfile = profile
    }
    
    private struct SimulatorState: Codable {
        let cash: Double
        let initialCapital: Double
        let trades: [Trade]
        let equityCurve: [PerformanceSnapshot]
        let positions: [Trade]
        let lastTradeTime: Date?
        let isEnabled: Bool
        let lastPriceUpdate: Double?
    }
}

// ============================================================================
// MARK: - 🆕 位置 2: Main App View (修改為整合 Simulator)
// ============================================================================
struct BTCTradingAppView: View {
    @StateObject private var api = BTCTradingAPI()
    @ObservedObject var simulator = TradingSimulator.shared  // 🆕 添加 Simulator
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // 🆕 傳入 simulator
            EnhancedDashboardView(api: api, simulator: simulator)
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
            
            // 🆕 傳入 simulator
            EnhancedSettingsView(api: api, simulator: simulator)
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(3)
        }
        .accentColor(.orange)
        .task {
            api.loadSignalHistory()
            await api.fetchPrice()
            await api.fetchSignal()
            
            // 🆕 處理信號
            if let signal = api.signal, let price = api.currentPrice {
                simulator.processSignal(signal: signal, currentPrice: price)
            }
        }
    }
}

// ============================================================================
// MARK: - 🆕 位置 3: Enhanced Dashboard with Trading
// ============================================================================
struct EnhancedDashboardView: View {
    @ObservedObject var api: BTCTradingAPI
    @ObservedObject var simulator: TradingSimulator
    @State private var autoRefresh = true
    let timer = Timer.publish(every: 300, on: .main, in: .common).autoconnect()
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // 原有價格卡片
                    PriceCard(
                        price: api.currentPrice,
                        change: api.priceChange24h
                    )
                    
                    // 🆕 AI 狀態卡片
                    AIStatusCard(
                        simulator: simulator,
                        signal: api.signal
                    )
                    
                    // 🆕 倉位狀態卡片
                    if !simulator.positions.isEmpty {
                        PositionsCard(
                            simulator: simulator,
                            currentPrice: api.currentPrice
                        )
                    }
                    
                    // 🆕 性能摘要
                    PerformanceSummaryCard(
                        simulator: simulator,
                        currentPrice: api.currentPrice
                    )
                    
                    // 原有信號卡片
                    if let signal = api.signal {
                        SignalSummaryCard(signal: signal)
                    }
                    
                    // 原有快速操作
                    QuickActionsCard(api: api)
                    
                    if let error = api.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding()
                    }
                }
                .padding()
            }
            .navigationTitle("BTC AI Trading")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        Task {
                            await refreshAll()
                        }
                    }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .onReceive(timer) { _ in
                if autoRefresh && simulator.isEnabled {
                    Task {
                        await refreshAll()
                    }
                }
            }
        }
        .task {
            await refreshAll()
        }
    }
    
    private func refreshAll() async {
        await api.fetchPrice()
        await api.fetchSignal()
        
        // 🆕 自動處理信號
        if let signal = api.signal, let price = api.currentPrice {
            simulator.processSignal(signal: signal, currentPrice: price)
        }
    }
}

// 🆕 AI 狀態卡片
struct AIStatusCard: View {
    @ObservedObject var simulator: TradingSimulator
    let signal: BTCTradingAPI.TradingSignal?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.purple)
                Text("AI Status")
                    .font(.headline)
                
                Spacer()
                
                Circle()
                    .fill(simulator.isEnabled ? Color.green : Color.gray)
                    .frame(width: 10, height: 10)
            }
            
            Divider()
            
            VStack(alignment: .leading, spacing: 8) {
                Text(simulator.aiPersonality)
                    .font(.subheadline)
                    .fontWeight(.bold)
                
                HStack {
                    Text("Auto Trading:")
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(simulator.isEnabled ? "Active ✅" : "Paused ⏸️")
                        .fontWeight(.semibold)
                        .foregroundColor(simulator.isEnabled ? .green : .gray)
                }
                
                if let signal = signal {
                    let threshold = simulator.calculatePersonalizedThreshold()
                    HStack {
                        Text("Threshold:")
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(threshold * 100))%")
                            .fontWeight(.semibold)
                            .foregroundColor(.blue)
                    }
                    
                    HStack {
                        Text("Decision:")
                            .foregroundColor(.secondary)
                        Spacer()
                        let adjustedConf = signal.confidence * (simulator.userProfile.tradingStyle == .aggressive ? 1.2 : simulator.userProfile.tradingStyle == .conservative ? 0.8 : 1.0)
                        let wouldTrade = adjustedConf >= threshold && simulator.canOpenNewPosition
                        Text(wouldTrade ? "Would Trade ✅" : "Would Skip ⏭️")
                            .fontWeight(.semibold)
                            .foregroundColor(wouldTrade ? .green : .orange)
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

// 🆕 倉位卡片
struct PositionsCard: View {
    @ObservedObject var simulator: TradingSimulator
    let currentPrice: Double?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "chart.bar.fill")
                    .foregroundColor(.orange)
                Text("Open Positions")
                    .font(.headline)
                
                Spacer()
                
                Text("\(simulator.positions.count)/\(simulator.maxPositions)")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.orange.opacity(0.2))
                    .cornerRadius(6)
            }
            
            Divider()
            
            ForEach(Array(simulator.positions.enumerated()), id: \.element.id) { index, position in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Position #\(index + 1)")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        
                        Spacer()
                        
                        if let price = currentPrice {
                            let pnl = (price - position.entryPrice) * position.quantity
                            let pnlPct = ((price - position.entryPrice) / position.entryPrice) * 100
                            
                            VStack(alignment: .trailing, spacing: 2) {
                                Text("$\(String(format: "%+.2f", pnl))")
                                    .font(.subheadline)
                                    .fontWeight(.bold)
                                    .foregroundColor(pnl >= 0 ? .green : .red)
                                Text("(\(String(format: "%+.2f", pnlPct))%)")
                                    .font(.caption)
                                    .foregroundColor(pnl >= 0 ? .green : .red)
                            }
                        }
                    }
                    
                    HStack {
                        Text("Entry: $\(String(format: "%.2f", position.entryPrice))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        if let price = currentPrice {
                            Text("Current: $\(String(format: "%.2f", price))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(Color.green.opacity(0.05))
                .cornerRadius(8)
            }
            
            if simulator.cooldownRemaining > 0 {
                HStack {
                    Image(systemName: "clock.fill")
                        .foregroundColor(.orange)
                    Text("Next trade in: \(simulator.cooldownText)")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
}

// 🆕 性能摘要卡片
struct PerformanceSummaryCard: View {
    @ObservedObject var simulator: TradingSimulator
    let currentPrice: Double?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Performance")
                .font(.headline)
            
            HStack(spacing: 15) {
                MetricBox(
                    label: "Equity",
                    value: "$\(String(format: "%.0f", simulator.currentEquity))",
                    color: .blue
                )
                
                MetricBox(
                    label: "Return",
                    value: String(format: "%+.1f%%", simulator.cumulativeReturn),
                    color: simulator.cumulativeReturn >= 0 ? .green : .red
                )
                
                MetricBox(
                    label: "Win Rate",
                    value: "\(Int(simulator.winRate * 100))%",
                    color: .purple
                )
            }
            
            HStack(spacing: 15) {
                MetricBox(
                    label: "Trades",
                    value: "\(simulator.totalTrades)",
                    color: .cyan
                )
                
                MetricBox(
                    label: "Profit Factor",
                    value: String(format: "%.2f", simulator.profitFactor),
                    color: .orange
                )
                
                MetricBox(
                    label: "Max DD",
                    value: String(format: "-%.1f%%", simulator.maxDrawdown),
                    color: .red
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
}

struct MetricBox: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.headline)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(color.opacity(0.1))
        .cornerRadius(8)
    }
}

// ============================================================================
// MARK: - 🆕 Enhanced Settings View
// ============================================================================
struct EnhancedSettingsView: View {
    @ObservedObject var api: BTCTradingAPI
    @ObservedObject var simulator: TradingSimulator
    @State private var showResetConfirmation = false
    @State private var showDiagnostic = false
    
    var body: some View {
        NavigationView {
            Form {
                // 🆕 AI 狀態部分
                Section(header: Text("AI Status")) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(simulator.aiPersonality)
                            .font(.headline)
                            .fontWeight(.bold)
                    }
                    
                    Toggle(isOn: $simulator.isEnabled) {
                        HStack {
                            Image(systemName: "chart.line.uptrend.xyaxis")
                                .foregroundColor(.orange)
                            Text("Auto Trading")
                        }
                    }
                    
                    HStack {
                        Text("Trading Style")
                        Spacer()
                        Text(simulator.userProfile.tradingStyle.rawValue)
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Risk Tolerance")
                        Spacer()
                        Text(String(format: "%.2f", simulator.userProfile.riskTolerance))
                            .foregroundColor(.secondary)
                    }
                    
                    // 🆕 閾值顯示
                    HStack {
                        Text("Current Threshold")
                        Spacer()
                        let threshold = simulator.calculatePersonalizedThreshold()
                        Text("\(Int(threshold * 100))%")
                            .fontWeight(.semibold)
                            .foregroundColor(.blue)
                    }
                }
                
                // 🆕 交易統計
                Section(header: Text("Trading Statistics")) {
                    HStack {
                        Text("Total Trades")
                        Spacer()
                        Text("\(simulator.totalTrades)")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Open Positions")
                        Spacer()
                        Text("\(simulator.positions.count)/\(simulator.maxPositions)")
                            .foregroundColor(simulator.positions.isEmpty ? .secondary : .green)
                    }
                    
                    HStack {
                        Text("Current Equity")
                        Spacer()
                        Text("$\(String(format: "%.2f", simulator.currentEquity))")
                            .foregroundColor(simulator.currentEquity >= simulator.initialCapital ? .green : .red)
                    }
                    
                    HStack {
                        Text("Win Rate")
                        Spacer()
                        Text("\(Int(simulator.winRate * 100))%")
                            .foregroundColor(.secondary)
                    }
                }
                
                // 🆕 診斷工具
                Section(header: Text("🔬 Edge AI Diagnostic")) {
                    Button(action: {
                        runDiagnostic()
                    }) {
                        HStack {
                            Image(systemName: "stethoscope")
                                .foregroundColor(.purple)
                            Text("Run Diagnostic")
                        }
                    }
                    
                    if showDiagnostic {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Diagnostic Results:")
                                .font(.caption)
                                .fontWeight(.bold)
                            
                            Text("Total Trades: \(simulator.totalTrades)")
                                .font(.caption)
                            Text("Win Rate: \(String(format: "%.1f", simulator.winRate * 100))%")
                                .font(.caption)
                            Text("Risk Tolerance: \(String(format: "%.2f", simulator.userProfile.riskTolerance))")
                                .font(.caption)
                            Text("Threshold: \(String(format: "%.4f", simulator.calculatePersonalizedThreshold()))")
                                .font(.caption)
                                .foregroundColor(.blue)
                            
                            Text("\n💡 Compare this with another device!")
                                .font(.caption2)
                                .foregroundColor(.orange)
                        }
                        .padding(8)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                    }
                }
                
                // 🆕 重置模擬器
                Section(header: Text("Simulator Management")) {
                    Button(role: .destructive, action: {
                        showResetConfirmation = true
                    }) {
                        HStack {
                            Image(systemName: "arrow.counterclockwise.circle.fill")
                            Text("Reset Simulator")
                        }
                    }
                    .confirmationDialog(
                        "Reset Trading Simulator?",
                        isPresented: $showResetConfirmation,
                        titleVisibility: .visible
                    ) {
                        Button("Reset to Aggressive") {
                            simulator.reset(preservePersonality: false, setStyle: .aggressive)
                        }
                        Button("Reset to Balanced") {
                            simulator.reset(preservePersonality: false, setStyle: .balanced)
                        }
                        Button("Reset to Conservative") {
                            simulator.reset(preservePersonality: false, setStyle: .conservative)
                        }
                        Button("Reset (Keep Personality)") {
                            simulator.reset(preservePersonality: true)
                        }
                        Button("Cancel", role: .cancel) {}
                    }
                }
                
                // 原有 Model 部分
                Section(header: Text("Model")) {
                    Button(action: {
                        Task {
                            await api.trainModel()
                        }
                    }) {
                        HStack {
                            Image(systemName: "brain")
                            Text("Retrain Model")
                        }
                    }
                }
                
                // 原有 Refresh 部分
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
                
                // 原有 About 部分
                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("2.0 (Edge AI)")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Architecture")
                        Spacer()
                        Text("Hybrid (Cloud + Edge)")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Initial Capital")
                        Spacer()
                        Text("$\(String(format: "%.0f", simulator.initialCapital))")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
    
    private func runDiagnostic() {
        showDiagnostic = true
        
        print("\n" + String(repeating: "=", count: 70))
        print("🔬 EDGE AI FULL DIAGNOSTIC")
        print(String(repeating: "=", count: 70))
        print("\n✅ Step 1: TradingSimulator exists")
        print("\n📊 Step 2: Device-specific data:")
        print("   Total Trades: \(simulator.totalTrades)")
        print("   Win Rate: \(String(format: "%.1f", simulator.winRate * 100))%")
        print("   Risk Tolerance: \(simulator.userProfile.riskTolerance)")
        print("   Trading Style: \(simulator.userProfile.tradingStyle.rawValue)")
        print("   AI Personality: \(simulator.aiPersonality)")
        
        let threshold = simulator.calculatePersonalizedThreshold()
        print("\n🎯 Step 3: Calculated threshold:")
        print("   Threshold: \(String(format: "%.4f", threshold))")
        
        print("\n🔍 Step 4: Edge AI Verification:")
        if simulator.totalTrades == 0 {
            print("   ⚠️  No trades yet - threshold is default (0.65)")
            print("   ✅ This is normal for new installation")
        } else {
            let deviation = abs(threshold - 0.65)
            if deviation > 0.01 {
                print("   ✅ Threshold adjusted by Edge AI (\(String(format: "%.2f", deviation * 100))%)")
                print("   ✅ Edge AI is working correctly!")
            } else {
                print("   ⚠️  Threshold unchanged - Edge AI may not be learning")
            }
        }
        
        print("\n" + String(repeating: "=", count: 70) + "\n")
    }
}

// ============================================================================
// MARK: - 原有 UI 組件 (保持不變)
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

struct QuickActionsCard: View {
    @ObservedObject var api: BTCTradingAPI

    var body: some View {
        VStack(spacing: 15) {
            Text("Quick Actions")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 15) {
                ActionButton(title: "Refresh", icon: "arrow.clockwise", color: .blue) {
                    Task {
                        await api.fetchPrice()
                        await api.fetchSignal()
                    }
                }

                ActionButton(title: "News", icon: "newspaper", color: .orange) {
                    Task {
                        await api.fetchNews()
                    }
                }

                ActionButton(title: "Train", icon: "brain", color: .purple) {
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

// Signal View (保持原樣)
struct SignalView: View {
    @ObservedObject var api: BTCTradingAPI
    
    var body: some View {
        NavigationView {
            Group {
                if let signal = api.signal {
                    ScrollView {
                        VStack(spacing: 20) {
                            DetailedSignalCard(signal: signal)
                            SentimentCard(
                                sentiment: signal.sentiment,
                                events: signal.events
                            )
                        }
                        .padding()
                    }
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
                    Button(action: {
                        Task {
                            await api.fetchSignal()
                        }
                    }) {
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
                DetailRow(label: "Current Price", value: "$\(String(format: "%.2f", signal.current_price))")
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

// MARK: - News View
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
                    Button(action: {
                        Task {
                            await api.fetchNews()
                        }
                    }) {
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
                HStack(spacing: 4) {
                    Circle()
                        .fill(sentimentColor)
                        .frame(width: 8, height: 8)
                    Text(sentimentText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
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

// MARK: - Settings View

struct SettingsView: View {
    @ObservedObject var api: BTCTradingAPI
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model")) {
                    Button(action: {
                        Task {
                            await api.trainModel()
                        }
                    }) {
                        HStack {
                            Image(systemName: "brain")
                            Text("Retrain Model")
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
        }
    }
}

// MARK: - Preview

struct BTCTradingAppView_Previews: PreviewProvider {
    static var previews: some View {
        BTCTradingAppView()
    }
}


