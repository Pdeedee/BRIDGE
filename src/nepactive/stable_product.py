import numpy as np
from scipy.optimize import linprog, milp
import random
import time

class MolecularSolverOptimized:
    def __init__(self, random_seed=None):
        # 如果提供了种子，使用它；否则使用当前时间
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        else:
            # 使用时间戳作为种子以确保每次运行都不同
            current_seed = int(time.time() * 1000) % 10000
            random.seed(current_seed)
            np.random.seed(current_seed)
            print(f"使用随机种子: {current_seed}")
        
        # 定义分子及其原子组成 [C, H, O, N]
        self.molecules = {
            'CH4': [1, 4, 0, 0],
            'CO': [1, 0, 1, 0],
            'CO2': [1, 0, 2, 0],
            'H2': [0, 2, 0, 0],
            'H2O': [0, 2, 1, 0],
            'N2': [0, 0, 0, 2],
            'NH3': [0, 3, 0, 1],
            'O2': [0, 0, 2, 0],
            'CHN': [1, 1, 0, 1],
            # 'CH2N2': [1, 2, 0, 2]
        }

        self.molecule_energies = {
            'CO': -14.747749,
            'NH3': -19.33091,
            'O2': -9.876768,
            'CHN': -19.782274,
            'CO2': -22.8497,
            'H2O': -14.086947,
            'CH4': -24.092106,
            'N2': -16.628838,
            'H2': -6.7345247,
            # 'CH2N2': -31.061832
        }
        
        # 创建原子-分子矩阵
        self.molecule_names = list(self.molecules.keys())
        self.atom_matrix = np.array([self.molecules[name] for name in self.molecule_names]).T
        # 创建能量数组（按分子顺序）
        self.energy_array = np.array([self.molecule_energies[name] for name in self.molecule_names])

    def verify_solution(self, solution, target_atoms):
        """验证解的正确性"""
        calculated_atoms = np.dot(self.atom_matrix, solution)
        error = np.sum(np.abs(calculated_atoms - target_atoms))
        
        print(f"解验证:")
        print(f"计算得到的原子: C={calculated_atoms[0]}, H={calculated_atoms[1]}, O={calculated_atoms[2]}, N={calculated_atoms[3]}")
        print(f"目标原子:       C={target_atoms[0]}, H={target_atoms[1]}, O={target_atoms[2]}, N={target_atoms[3]}")
        print(f"差值:           C={target_atoms[0]-calculated_atoms[0]}, H={target_atoms[1]-calculated_atoms[1]}, O={target_atoms[2]-calculated_atoms[2]}, N={target_atoms[3]-calculated_atoms[3]}")
        print(f"总误差: {error}")
        
        return error

    def energy_minimization_solution(self, c, h, o, n):
        """使用能量最小化的整数线性规划"""
        target_atoms = np.array([c, h, o, n])
        A_eq = self.atom_matrix
        b_eq = target_atoms
        
        # 定义变量界限 (非负整数)
        bounds = [(0, None) for _ in range(len(self.molecules))]
        
        # 目标函数：最小化总能量
        c_obj = self.energy_array.copy()
        
        # 解整数线性规划问题
        try:
            integrality = np.ones(len(self.molecules))  # 全部变量都是整数
            result = milp(c=c_obj, constraints=[
                {"A": A_eq, "b": b_eq, "type": "=="}
            ], integrality=integrality, bounds=bounds)
            
            if result.success:
                solution = result.x.astype(int)
                error = self.verify_solution(solution, target_atoms)
                return solution, error
            else:
                print("精确求解失败，尝试近似方法")
                return self.improved_approximate_solution(c, h, o, n)
        except Exception as e:
            print(f"精确求解出错: {e}")
            return self.improved_approximate_solution(c, h, o, n)

    def improved_approximate_solution(self, c, h, o, n):
        """改进的近似求解方法"""
        target_atoms = np.array([c, h, o, n])
        num_atoms = 4
        num_molecules = len(self.molecules)
        
        # 构建优化问题 - 多次尝试不同的权重
        best_solution = None
        best_error = float('inf')
        
        for attempt in range(10):  # 多次尝试
            # 构建目标函数
            c_obj = np.zeros(num_molecules + 2 * num_atoms)
            
            # 能量项 + 随机扰动
            energy_weight = 0.01 * (1 + random.random())
            c_obj[:num_molecules] = energy_weight * self.energy_array + np.random.rand(num_molecules) * 0.001
            
            # 误差惩罚项
            error_weight = 1000 * (1 + random.random())
            c_obj[num_molecules:] = error_weight
            
            # 约束条件：A_eq * x + e_plus - e_minus = b_eq
            A_eq = np.zeros((num_atoms, num_molecules + 2 * num_atoms))
            A_eq[:, :num_molecules] = self.atom_matrix
            for i in range(num_atoms):
                A_eq[i, num_molecules + i] = 1           # e_plus[i]
                A_eq[i, num_molecules + num_atoms + i] = -1  # e_minus[i]
            
            b_eq = target_atoms
            bounds = [(0, None) for _ in range(num_molecules + 2 * num_atoms)]
            
            # 解线性规划问题
            try:
                result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                
                if result.success:
                    # 提取分子数量并四舍五入
                    solution = np.zeros(num_molecules, dtype=int)
                    for i in range(num_molecules):
                        val = result.x[i]
                        # 智能四舍五入
                        if val < 0.1:
                            solution[i] = 0
                        else:
                            solution[i] = int(round(val))
                    
                    error = self.verify_solution(solution, target_atoms)
                    
                    if error < best_error:
                        best_error = error
                        best_solution = solution
                        
                        if error == 0:  # 找到完美解
                            break
            except Exception as e:
                print(f"线性规划尝试 {attempt+1} 失败: {e}")
                continue
        
        if best_solution is not None:
            return best_solution, best_error
        else:
            return self.fixed_random_solution(c, h, o, n)

    def fixed_random_solution(self, c, h, o, n, num_tries=10000):
        """修复了bug的随机解方法"""
        target_atoms = np.array([c, h, o, n])
        best_solution = None
        best_error = float('inf')
        
        print(f"开始随机搜索，尝试 {num_tries} 次...")
        
        for attempt in range(num_tries):
            # 生成随机解
            solution = np.zeros(len(self.molecules), dtype=int)
            remaining = target_atoms.copy()
            
            # 随机顺序添加分子
            mol_indices = list(range(len(self.molecules)))
            random.shuffle(mol_indices)
            
            for mol_idx in mol_indices:
                mol_atoms = self.atom_matrix[:, mol_idx]
                
                # 计算可以添加的最大分子数量 - 修复了这里的bug！
                if np.any(mol_atoms > 0):
                    max_possible = min(
                        remaining[i] // mol_atoms[i] 
                        for i in range(4) if mol_atoms[i] > 0
                    )
                else:
                    max_possible = 0
                
                # 随机选择添加数量，但倾向于添加更多
                if max_possible > 0:
                    # 使用加权随机，倾向于选择较大的数量
                    weights = [(i+1)**0.5 for i in range(max_possible + 1)]
                    total_weight = sum(weights)
                    r = random.random() * total_weight
                    
                    amount = 0
                    cumulative = 0
                    for i, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            amount = i
                            break
                    
                    solution[mol_idx] = amount
                    remaining -= mol_atoms * amount
            
            # 计算误差
            error = np.sum(np.abs(remaining))
            if error < best_error:
                best_error = error
                best_solution = solution
                
                if attempt % 1000 == 0:
                    print(f"尝试 {attempt}: 当前最佳误差 = {best_error}")
                
                # 如果找到完美解，直接返回
                if error == 0:
                    print(f"在第 {attempt+1} 次尝试中找到完美解!")
                    break
        
        if best_solution is not None:
            self.verify_solution(best_solution, target_atoms)
        
        return best_solution, best_error

    def iterative_improvement(self, c, h, o, n, max_iterations=100):
        """迭代改进方法"""
        target_atoms = np.array([c, h, o, n])
        
        # 从一个初始解开始
        current_solution, current_error = self.fixed_random_solution(c, h, o, n, 1000)
        
        if current_error == 0:
            return current_solution, current_error
            
        print(f"开始迭代改进，初始误差: {current_error}")
        
        for iteration in range(max_iterations):
            improved = False
            
            # 尝试调整每个分子的数量
            for mol_idx in range(len(self.molecules)):
                original_count = current_solution[mol_idx]
                
                # 尝试增加或减少这个分子的数量
                for delta in [-2, -1, 1, 2]:
                    new_count = max(0, original_count + delta)
                    
                    # 创建新解
                    new_solution = current_solution.copy()
                    new_solution[mol_idx] = new_count
                    
                    # 计算新误差
                    calculated_atoms = np.dot(self.atom_matrix, new_solution)
                    new_error = np.sum(np.abs(calculated_atoms - target_atoms))
                    
                    # 如果更好，接受这个解
                    if new_error < current_error:
                        current_solution = new_solution
                        current_error = new_error
                        improved = True
                        print(f"迭代 {iteration+1}: 改进到误差 {current_error}")
                        
                        if current_error == 0:
                            self.verify_solution(current_solution, target_atoms)
                            return current_solution, current_error
            
            # 如果没有改进，尝试同时调整两个分子
            if not improved:
                for i in range(len(self.molecules)):
                    for j in range(i+1, len(self.molecules)):
                        for delta_i in [-1, 1]:
                            for delta_j in [-1, 1]:
                                new_solution = current_solution.copy()
                                new_solution[i] = max(0, new_solution[i] + delta_i)
                                new_solution[j] = max(0, new_solution[j] + delta_j)
                                
                                calculated_atoms = np.dot(self.atom_matrix, new_solution)
                                new_error = np.sum(np.abs(calculated_atoms - target_atoms))
                                
                                if new_error < current_error:
                                    current_solution = new_solution
                                    current_error = new_error
                                    improved = True
                                    print(f"迭代 {iteration+1}: 双重调整改进到误差 {current_error}")
                                    
                                    if current_error == 0:
                                        self.verify_solution(current_solution, target_atoms)
                                        return current_solution, current_error
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # 如果还是没有改进，就停止
            if not improved:
                print(f"在迭代 {iteration+1} 无法继续改进")
                break
        
        self.verify_solution(current_solution, target_atoms)
        return current_solution, current_error

    def solve(self, c, h, o, n, method=None):
        """使用指定方法找到解"""
        print(f"\n=== 求解 C={c}, H={h}, O={o}, N={n} ===")
        
        if method == 'energy':
            print("使用能量最小化方法")
            return self.energy_minimization_solution(c, h, o, n)
        elif method == 'fixed_random':
            print("使用修复后的随机方法")
            return self.fixed_random_solution(c, h, o, n)
        elif method == 'iterative':
            print("使用迭代改进方法")  
            return self.iterative_improvement(c, h, o, n)
        else:
            # 默认尝试多种方法
            methods = [
                ('energy', lambda: self.energy_minimization_solution(c, h, o, n)),
                ('iterative', lambda: self.iterative_improvement(c, h, o, n, 50)),
                ('fixed_random', lambda: self.fixed_random_solution(c, h, o, n, 5000))
            ]
            
            best_solution = None
            best_error = float('inf')
            best_method = None
            
            for name, method_func in methods:
                print(f"\n--- 尝试方法: {name} ---")
                try:
                    solution, error = method_func()
                    if error < best_error:
                        best_error = error
                        best_solution = solution  
                        best_method = name
                        
                    if error == 0:
                        print(f"方法 {name} 找到完美解!")
                        break
                        
                except Exception as e:
                    print(f"方法 {name} 失败: {e}")
                    continue
            
            print(f"\n最佳方法: {best_method}, 误差: {best_error}")
            return best_solution, best_error
    
    def format_solution(self, solution):
        """格式化解的输出"""
        result = {}
        for i, molecule in enumerate(self.molecule_names):
            if solution[i] > 0:
                result[molecule] = solution[i]
        return result
    
    def calculate_atoms(self, solution):
        """计算给定解所使用的原子数量"""
        return np.dot(self.atom_matrix, solution)
    
    def calculate_total_energy(self, solution):
        """计算给定解的总能量"""
        return np.dot(solution, self.energy_array)


def solve_molecular_distribution(c, h, o, n, method=None, random_seed=None):
    """对外暴露的主函数"""
    solver = MolecularSolverOptimized(random_seed)
    solution, error = solver.solve(c, h, o, n, method)
    
    if solution is None:
        return {
            "solution": {},
            "used_atoms": {"C": 0, "H": 0, "O": 0, "N": 0},
            "target_atoms": {"C": c, "H": h, "O": o, "N": n},
            "error": float('inf'),
            "total_energy": 0
        }
    
    total_energy = solver.calculate_total_energy(solution)
    
    result = {
        "solution": solver.format_solution(solution),
        "used_atoms": {
            "C": solver.calculate_atoms(solution)[0],
            "H": solver.calculate_atoms(solution)[1],
            "O": solver.calculate_atoms(solution)[2],
            "N": solver.calculate_atoms(solution)[3]
        },
        "target_atoms": {"C": c, "H": h, "O": o, "N": n},
        "error": error,
        "total_energy": total_energy
    }
    
    return result

# 测试修复后的版本
if __name__ == "__main__":
    # 测试您的例子
    c, h, o, n = 24, 48, 12, 48
    
    print("=== 测试修复后的求解器 ===")
    result = solve_molecular_distribution(c, h, o, n, method='energy')
    print(f"\n最终结果:")
    print(f"解: {result['solution']}")
    print(f"使用的原子: {result['used_atoms']}")
    print(f"目标原子: {result['target_atoms']}")
    print(f"总能量: {result['total_energy']:.6f}")
    print(f"误差: {result['error']}")