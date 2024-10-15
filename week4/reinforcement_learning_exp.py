

# Reinforcement learning (pekiştirmeli öğrenme), bir ajan (agent) ile çevresi (environment) arasında etkileşimlerin olduğu bir makine öğrenimi tekniğidir. Ajan, belirli bir görevi yerine getirirken çevresinden sürekli geri bildirim alır ve bu geri bildirimler doğrultusunda gelecekteki davranışlarını iyileştirir. Amaç, uzun vadede en fazla ödülü (reward) elde edecek bir politika (policy) öğrenmektir.

# Reinforcement Learning’in Temel Bileşenleri:
# Ajan (Agent): Öğrenen ve aksiyonları gerçekleştiren birimdir. Ajanın görevi, çevreden maksimum ödül elde etmektir.
# Çevre (Environment): Ajanın etkileşimde bulunduğu ortamdır. Ajan, bu ortamda eylemler gerçekleştirir ve çevre, geri bildirim olarak ödül veya ceza verir.
# Durum (State): Ajanın içinde bulunduğu çevredeki mevcut durumu tanımlayan bilgidir.
# Aksiyon (Action): Ajanın mevcut duruma göre yapabileceği eylemler.
# Ödül (Reward): Ajanın bir aksiyondan sonra aldığı geri bildirim. Ajanın ödülü maksimuma çıkarmak için strateji geliştirmesi beklenir.
# Politika (Policy): Ajanın her durumda nasıl bir aksiyon seçeceğini belirleyen strateji.
# Q-Değeri (Q-Value): Ajanın belli bir duruma göre belli bir aksiyonun ne kadar ödül getirebileceğini gösteren bir değer.
# Reinforcement learning’de ajan, ödüllere göre stratejisini geliştirir ve en uygun aksiyonları zamanla öğrenir. Bu teknik, robotik, oyunlar ve otonom sistemler gibi birçok alanda kullanılır.

# Python’da Q-Learning Örneği
# Q-Learning, yaygın bir reinforcement learning algoritmasıdır. Bu algoritmada, ajan her duruma göre aksiyonların ödüllerini öğrenir ve ödülü maksimuma çıkaracak aksiyonları seçer.

# Aşağıda bir gridworld ortamında Q-Learning algoritmasının bir uygulaması gösterilmiştir. Ajanın amacı, başlangıç noktasından ödül noktasına en kısa yoldan ulaşmaktır.


import numpy as np
import random

# Gridworld ortamı (5x5 boyutunda bir grid)
environment = np.zeros((5, 5))

# Ödül noktası (4, 4)
environment[4, 4] = 10  # Ödül

# Hareket seçenekleri: yukarı, aşağı, sola, sağa
actions = ['up', 'down', 'left', 'right']

# Q-table (Q-değerleri saklanacak)
Q_table = np.zeros((5, 5, len(actions)))

# Öğrenme parametreleri
alpha = 0.1  # Öğrenme hızı
gamma = 0.9  # Gelecekteki ödüllere ne kadar değer verdiğimizi belirler
epsilon = 0.8  # Keşfetme oranı (başlangıçta keşfetmeye daha çok odaklanır)

# Ajanın başlangıç pozisyonu
agent_position = [0, 0]

def choose_action(position):
    """Epsilon-greedy stratejisiyle aksiyon seçimi."""
    if random.uniform(0, 1) < epsilon:
        # Rastgele aksiyon (keşfetme)
        return random.choice(actions)
    else:
        # En iyi aksiyon (kazanma)
        state_actions = Q_table[position[0], position[1], :]
        return actions[np.argmax(state_actions)]

def take_action(position, action):
    """Verilen aksiyona göre ajanı hareket ettirme."""
    new_position = position.copy()
    
    if action == 'up' and position[0] > 0:
        new_position[0] -= 1
    elif action == 'down' and position[0] < 4:
        new_position[0] += 1
    elif action == 'left' and position[1] > 0:
        new_position[1] -= 1
    elif action == 'right' and position[1] < 4:
        new_position[1] += 1
    
    return new_position

def get_reward(position):
    """Ajanın ödülünü hesaplama."""
    if position == [4, 4]:
        return 10
    else:
        return -0.1

# Q-Learning algoritması
for episode in range(1000):
    agent_position = [0, 0]  # Ajan her seferinde başlangıç noktasından başlar
    
    while agent_position != [4, 4]:  # Ödül noktasına ulaşana kadar
        action = choose_action(agent_position)
        new_position = take_action(agent_position, action)
        
        reward = get_reward(new_position)
        
        # Q-değerini güncelleme
        old_q_value = Q_table[agent_position[0], agent_position[1], actions.index(action)]
        future_q_value = np.max(Q_table[new_position[0], new_position[1], :])
        
        Q_table[agent_position[0], agent_position[1], actions.index(action)] = \
            old_q_value + alpha * (reward + gamma * future_q_value - old_q_value)
        
        agent_position = new_position  # Ajan yeni pozisyona geçer

# Sonuç Q-Tablosunu gösterme
print("Q-table:")
print(Q_table)
