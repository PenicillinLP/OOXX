import numpy as np  # 做题先写 “解”

# 因为我们之后需要创建两个 Agent 互相下棋，所以定义一个 Agent 的类会方便一点
class Agent():

    # 初始化，不复杂，参数只有仨
    def __init__(self,OOXX_Index,Epsilon,Alpha):
        self.index = OOXX_Index # OOXX_Index 用 1 或者 2 代表是两个 Agent 当中的哪一个
        self.epsilon = Epsilon # Epsilon 就是 ε-Greedy 策略中的随机选择概率
        self.alpha = Alpha # Alpha 就是学习率
        self.value = np.zeros((3,3,3,3,3,3,3,3,3)) # 储存状态价值的表
        # OOXX的棋盘一共有 9 个位置，每个位置有 3 种情况（O、X、无）
        # 所以我们用一个 9 维的向量来表示状态，每个维度表示一个位置上的 3 种情况
        # 想像一下我们用一个 361 维向量来表示围棋的状态，这个向量会非常巨大
        # 在实际上不可行，所以对于状态空间非常大的情况我们需要别的表示状态的方法
        # 例如使用深度神经网络 （具体可以参考我讲解 AlphaGo 的视频 https://www.bilibili.com/video/BV1hb4y197he）
        self.stored_Outcome = np.zeros(9).astype(np.int8) # Agent 内部记录的后果，初始化为 0，表示空棋盘
        # 因为我们要将后果作为价值矩阵的索引，所以用 .astype(np.int8) 规定为整数

    # 重置状态
    def reset(self):
        self.stored_Outcome = np.zeros(9).astype(np.int8)

    # 输入为状态，输出为后果，同时进行价值更新
    def move(self,State):
        Outcome = State.copy() # 拷贝一份状态
        # 因为这个状态 State 对另一名玩家来说是后果
        # 需要留着用来学习价值，所以不能直接更改，因此 .copy() 拷贝一份
        available = np.where(Outcome==0)[0] # 先判断棋盘上有哪些地方可以落子，也就是 Outcome==0 的地方
        if np.random.binomial(1,self.epsilon): # 判断要不要采取 ε-Greedy 的随机行动
            Outcome[np.random.choice(available)] = self.index # 随机选择一个位置标注为 1 或着 2 （取决于是 Agent1 还是 Agent2）
        else: # 如果不随机，就采用最优策略
            temp_Value = np.zeros(len(available)) # 创建一个临时的价值向量
            for i in range(len(available)): # 对每一个可能落子的地方
                temp_Outcome = Outcome.copy() # 拷贝当前时刻的状态
                temp_Outcome[available[i]] = self.index # 假设在一个地方落子，得到后果
                temp_Value[i] = self.value[tuple(temp_Outcome)] # 调用价值函数，计算得到的后果的价值
            choose = np.argmax(temp_Value) # 选择价值最大的那一个行动
            Outcome[available[choose]] = self.index # 把选择的那个位置标注为 1 或着 2 （取决于是 Agent1 还是 Agent2）
        # 基于误差的学习法，或者说就是时序差分法 （在这个例子中即时奖励为 0）
        Error = self.value[tuple(Outcome)] - self.value[tuple(self.stored_Outcome)]
        #计算当前的后果的价值估计和储存的（上一个）后果的价值估计的误差
        self.value[tuple(self.stored_Outcome)] += self.alpha*Error # 更新储存的（上一个）后果的价值估计
        self.stored_Outcome = Outcome.copy() # 把当前的后果储存（因为接着就要进行下一步了）
        return Outcome # 返回当前的后果

# 写一个函数判断输赢
# 这里我们直接用暴力枚举判断输赢就行了，因为状态空间就这么简单
def Judge(Outcome, curPlayer): # 输入为状态和对应的玩家
    Triple = np.repeat(curPlayer.index,3)
    winner = 0  # 默认胜负未分
    if 0 not in Outcome: # 没地方下了
        winner = 3 # 平局
    if (Outcome[0:3]==Triple).all() or (Outcome[3:6]==Triple).all() or (Outcome[6:9]==Triple).all(): # 分别判断三行
        winner = curPlayer.index
    if (Outcome[0:7:3]==Triple).all() or (Outcome[1:8:3]==Triple).all() or (Outcome[2:9:3]==Triple).all(): # 分别判断三列
        winner = curPlayer.index
    if (Outcome[0:9:4]==Triple).all() or (Outcome[2:7:2]==Triple).all(): # 分别判断两条对角线
        winner = curPlayer.index
    return winner # 返回玩家是否胜利

# 创建两个 Agent
# 大家可以自己改变参数
Agent1 = Agent(OOXX_Index = 1,Epsilon=0.1,Alpha=0.1)
Agent2 = Agent(OOXX_Index = 2,Epsilon=0.1,Alpha=0.1)

Trial = 30000 # 训练 3 万次
Winner = np.zeros(Trial) # 记录结果

for i in range(Trial):
    if i==20000:  # 在 2 万次之后取消掉随机性
        Agent1.epsilon = 0
        Agent2.epsilon = 0
    Agent1.reset() # 重置状态
    Agent2.reset() # 重置状态
    winner = 0 # 默认胜负未分
    State = np.zeros(9).astype(np.int8) # 初始化棋盘
    # 我们默认 Agent1 先行
    # 并且以 Agent1 的视角定义 State 和 Outcome
    curPlayer = Agent1
    opponentPlayer = Agent2
    while winner == 0: #如果胜负未分
        Outcome = curPlayer.move(State) # 当前 Agent 采取行动，并且更新价值
        winner = Judge(Outcome,curPlayer) # 判断当前 Agent 是否获胜
        if winner == curPlayer.index: # 如果当前 Agent 获胜
            curPlayer.value[tuple(Outcome)] = 1 # Outcome 的价值对当前 Agent 来说为 1
            opponentPlayer.value[tuple(State)] = -1 # 对手Agent 对应的后果，也就是当前 Agent 面临的 State 的价值对 对方 Agent来说为 -1
            
        curPlayer, opponentPlayer = opponentPlayer, curPlayer # 下一轮对手下棋，交换角色
        State = Outcome # 更新下一轮棋盘状态
        
    Winner[i] = winner # 记录结果


import matplotlib.pyplot as plt

# 根据结果计算胜率
step = 250 # 每隔250局游戏计算一次胜率
duration = 500 # 胜率根据前后共500局来计算
def Rate(Winner):
    Rate1 = np.zeros(int((Trial-duration)/step)+1) # Agent1 胜率
    Rate2 = np.zeros(int((Trial-duration)/step)+1) # Agent2 胜率
    Rate3 = np.zeros(int((Trial-duration)/step)+1) # 平局概率
    for i in range(len(Rate1)):
        Rate1[i] = np.sum(Winner[step*i:duration+step*i]==1)/duration
        Rate2[i] = np.sum(Winner[step*i:duration+step*i]==2)/duration
        Rate3[i] = np.sum(Winner[step*i:duration+step*i]==3)/duration
    return Rate1,Rate2,Rate3

Rate1,Rate2,Rate3=Rate(Winner)

fig,ax=plt.subplots(figsize=(16,9))
plt.plot(Rate1,linewidth=4,marker='.',markersize=20,color="#0071B7",label="Agent1")
plt.plot(Rate2,linewidth=4,marker='.',markersize=20,color="#DB2C2C",label="Agent2")
plt.plot(Rate3,linewidth=4,marker='.',markersize=20,color="#FAB70D",label="Draw")
plt.xticks(np.arange(0,121,40),np.arange(0,31+1,10),fontsize=30)
plt.yticks(np.arange(0,1.1,0.2),np.round(np.arange(0,1.1,0.2),2),fontsize=30)
plt.xlabel("Trials(x1k)",fontsize=30)
plt.ylabel("Winning Rate",fontsize=30)
plt.legend(loc="best",fontsize=25)
plt.tick_params(width=4,length=10)
ax.spines[:].set_linewidth(4)
plt.savefig('ooxx_prime_plot.png', dpi=300, bbox_inches='tight')