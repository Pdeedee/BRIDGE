import numpy as np
from ase import io, Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from collections import defaultdict, Counter
from typing import List, Dict, Set
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform

def identify_molecules_in_frame(atoms, mult_factor=0.7) -> List[Dict]:
    """
    识别每一帧中的分子，并返回分子信息和原子索引
    增加了截断半径系数，提高成键检测的敏感性
    """
    visited = set()  # 用于记录已访问的原子索引
    molecules = []   # 用于存储识别到的分子

    # 基于共价半径为每个原子生成径向截止
    cutoffs = natural_cutoffs(atoms, mult=mult_factor)
    
    # 获取成键原子，考虑周期性边界条件
    nl = NeighborList(cutoffs=cutoffs, bothways=True, self_interaction=False)
    nl.update(atoms)  # 更新邻居列表
    
    # 遍历所有原子
    for i in range(len(atoms)):
        if i not in visited:  # 如果当前原子尚未被访问
            current_molecule = defaultdict(int)  # 用于统计元素及其数量
            molecule_atoms_indices = []  # 用于记录属于该分子的原子索引
            
            stack = [i]  # 使用栈进行深度优先搜索，初始化栈为当前原子索引
            while stack:
                atom_index = stack.pop()  # 从栈中取出一个原子索引
                if atom_index not in visited:
                    visited.add(atom_index)  # 标记为已访问
                    molecule_atoms_indices.append(atom_index)  # 记录属于该分子的原子索引
                    
                    atom_symbol = atoms[atom_index].symbol  # 获取原子的元素符号
                    current_molecule[atom_symbol] += 1  # 统计该元素的数量
                    
                    # 获取与当前原子成键的原子索引
                    bonded_indices, _ = nl.get_neighbors(atom_index)
                    stack.extend(idx for idx in bonded_indices if idx not in visited)

            # 将当前分子信息和原子索引添加到分子列表中
            if current_molecule:
                molecule_name = ''.join(f"{element}{current_molecule[element]}" for element in sorted(current_molecule.keys()))
                molecules.append({
                    'formula': molecule_name,
                    'composition': dict(current_molecule),
                    'atom_indices': molecule_atoms_indices
                })

    return molecules



def get_minimum_image_vector(vector, cell):
    """
    计算考虑周期性边界条件的最小映像向量
    """
    # 转换到分数坐标
    scaled_vector = np.linalg.solve(cell.T, vector.T).T
    
    # 应用最小映像约定
    scaled_vector -= np.round(scaled_vector)
    
    # 转回笛卡尔坐标
    minimum_vector = np.dot(scaled_vector, cell)
    
    return minimum_vector

def extract_molecule_with_bonds(atoms, atom_indices, nl, debug=False):
    """
    提取分子并正确处理周期性边界条件下的键
    返回一个新的Atoms对象，其中原子已经被适当地重新定位以保持分子的完整性
    改进了offset的应用逻辑
    """
    if not atom_indices:
        return None
    
    # 创建一个新的原子列表，用于存储重新定位后的原子
    new_positions = []
    new_symbols = []
    atom_to_new_index = {}  # 用于跟踪原始原子索引到新位置的映射
    
    # 获取盒子大小，用于调试输出
    cell = atoms.get_cell()
    if debug:
        print(f"Cell dimensions: {np.diag(cell)}")
    
    # 选择第一个原子作为参考点
    reference_index = atom_indices[0]
    reference_pos = atoms.positions[reference_index].copy()
    
    # 添加参考原子到新列表
    new_symbols.append(atoms[reference_index].symbol)
    new_positions.append(reference_pos)
    atom_to_new_index[reference_index] = 0
    
    # 使用广度优先搜索来构建分子，确保所有原子都相对于参考原子正确定位
    visited = {reference_index}
    queue = [reference_index]
    
    if debug:
        print(f"Starting BFS from atom {reference_index} ({atoms[reference_index].symbol})")
        print(f"Reference position: {reference_pos}")
    
    while queue:
        current_index = queue.pop(0)
        current_pos = new_positions[atom_to_new_index[current_index]]
        
        # 获取与当前原子成键的所有原子
        bonded_indices, offsets = nl.get_neighbors(current_index)
        
        if debug and len(bonded_indices) > 0:
            print(f"Atom {current_index} has {len(bonded_indices)} neighbors")
            print(f"Bonded indices: {bonded_indices}")
            print(f"Offsets: {offsets}")
        
        for neighbor_index, offset in zip(bonded_indices, offsets):
            if neighbor_index in atom_indices and neighbor_index not in visited:
                # 标记为已访问
                visited.add(neighbor_index)
                queue.append(neighbor_index)
                
                # 获取邻居原子的元素符号
                neighbor_symbol = atoms[neighbor_index].symbol
                
                # 原始位置
                original_pos = atoms.positions[neighbor_index]
                
                # 计算考虑周期性边界条件的正确位置
                # 改进：使用最小映像原则直接计算
                direct_vector = original_pos - atoms.positions[current_index]
                # 将向量转换到分数坐标
                fractional_vector = np.linalg.solve(cell.T, direct_vector.T).T
                # 应用最小映像约定
                fractional_vector -= np.round(fractional_vector)
                # 转回笛卡尔坐标
                min_image_vector = np.dot(fractional_vector, cell)
                
                # 使用当前原子的新位置加上最小映像向量
                correct_pos = current_pos + min_image_vector
                
                # 为确保一致性，也计算使用offset的位置用于比较
                offset_pos = original_pos + np.dot(offset, cell)
                
                if debug:
                    print(f"  Neighbor {neighbor_index} ({neighbor_symbol})")
                    print(f"  Original position: {original_pos}")
                    print(f"  Current atom (original): {atoms.positions[current_index]}")
                    print(f"  Current atom (new): {current_pos}")
                    print(f"  Offset: {offset}")
                    print(f"  Offset vector: {np.dot(offset, cell)}")
                    print(f"  Position using offset: {offset_pos}")
                    print(f"  Direct vector: {direct_vector}")
                    print(f"  Fractional vector: {fractional_vector}")
                    print(f"  Min image vector: {min_image_vector}")
                    print(f"  Position using min image: {correct_pos}")
                    print(f"  Distance (original): {np.linalg.norm(direct_vector)}")
                    print(f"  Distance (offset): {np.linalg.norm(offset_pos - current_pos)}")
                    print(f"  Distance (min image): {np.linalg.norm(min_image_vector)}")
                
                # 使用最小映像向量计算的位置
                new_positions.append(correct_pos)
                new_symbols.append(neighbor_symbol)
                atom_to_new_index[neighbor_index] = len(new_positions) - 1
    
    # 检查是否所有原子都被访问到了
    if len(visited) != len(atom_indices):
        missing_atoms = set(atom_indices) - visited
        if debug:
            print(f"Warning: Not all atoms were visited. Missing atoms: {missing_atoms}")
        
        # 尝试使用邻居列表的增强版本
        missing_found = 0
        
        # 对于每个未访问的原子，找到最近的已访问原子作为参考
        for missing_idx in list(missing_atoms):
            if missing_idx in visited:  # 可能在前面的迭代中已经被添加
                continue
                
            # 查找所有已访问原子与当前未访问原子的距离
            min_dist = float('inf')
            closest_idx = None
            
            # 在原始坐标系中计算距离
            for visited_idx in visited:
                dist_vector = atoms.positions[missing_idx] - atoms.positions[visited_idx]
                
                # 应用最小映像约定
                fractional_vector = np.linalg.solve(cell.T, dist_vector.T).T
                fractional_vector -= np.round(fractional_vector)
                min_image_vector = np.dot(fractional_vector, cell)
                
                dist = np.linalg.norm(min_image_vector)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = visited_idx
            
            if closest_idx is not None and closest_idx in atom_to_new_index:
                # 从找到的最近原子计算正确位置
                ref_new_pos = new_positions[atom_to_new_index[closest_idx]]
                ref_orig_pos = atoms.positions[closest_idx]
                missing_pos = atoms.positions[missing_idx]
                
                # 计算从参考原子到缺失原子的最小映像向量
                vector = missing_pos - ref_orig_pos
                fractional_vector = np.linalg.solve(cell.T, vector.T).T
                fractional_vector -= np.round(fractional_vector)
                min_image_vector = np.dot(fractional_vector, cell)
                
                # 新位置 = 参考原子的新位置 + 最小映像向量
                corrected_pos = ref_new_pos + min_image_vector
                
                # 添加到分子中
                new_symbols.append(atoms[missing_idx].symbol)
                new_positions.append(corrected_pos)
                atom_to_new_index[missing_idx] = len(new_positions) - 1
                visited.add(missing_idx)
                missing_found += 1
                
                if debug:
                    print(f"Added missing atom {missing_idx} using nearest neighbor {closest_idx}")
                    print(f"  Original position: {missing_pos}")
                    print(f"  Corrected position: {corrected_pos}")
                    print(f"  Distance to reference: {np.linalg.norm(min_image_vector)}")
        
        if debug and missing_found > 0:
            print(f"Successfully added {missing_found} missing atoms")
        
        # 再次检查是否所有原子都被访问到了
        if len(visited) != len(atom_indices):
            still_missing = set(atom_indices) - visited
            if debug:
                print(f"Warning: Still missing {len(still_missing)} atoms: {still_missing}")
    
    # 创建新的Atoms对象
    molecule = Atoms(symbols=new_symbols, positions=new_positions)
    
    # 将分子移动到盒子中心
    molecule.center()
    
    return molecule

def save_unique_molecules_as_pdb(trajectory_file: str, output_dir: str = 'unique_molecules', index=':', mult_factor=0.7, debug=False):
    """
    从轨迹文件中提取不重复的分子并保存为PDB文件，支持单帧和多帧文件
    正确处理周期性边界条件
    
    参数:
        trajectory_file: 轨迹文件路径
        output_dir: 输出目录
        index: 轨迹索引
        mult_factor: 截断半径系数，影响成键检测的灵敏度
        debug: 是否输出调试信息
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建一个子目录用于存放有问题的分子
    problem_dir = os.path.join(output_dir, "problem_molecules")
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)
    
    # 读取轨迹文件
    try:
        # 先尝试读取为多帧轨迹
        traj = io.read(trajectory_file, index=index)
        
        # 检查是否为列表（多帧）或单个Atoms对象（单帧）
        if not isinstance(traj, list):
            traj = [traj]  # 将单帧转换为列表格式统一处理
    except Exception as e:
        print(f"Error reading trajectory: {e}")
        # 尝试作为单帧文件读取
        try:
            atoms = io.read(trajectory_file)
            traj = [atoms]
        except Exception as e2:
            print(f"Could not read file as a trajectory or a single frame: {e2}")
            return []
    
    # 用于记录已保存的分子类型
    saved_molecules = set()
    saved_molecules_info = {}
    problem_molecules = {}
    
    # 遍历轨迹的每一帧
    for frame_idx, atoms in enumerate(traj):
        print(f"Analyzing frame {frame_idx + 1} of {len(traj)}...")
        
        # 基于共价半径为每个原子生成径向截止
        cutoffs = natural_cutoffs(atoms, mult=mult_factor)
        
        # 获取成键原子，考虑周期性边界条件
        nl = NeighborList(cutoffs=cutoffs, bothways=True, self_interaction=False)
        nl.update(atoms)  # 更新邻居列表
        
        # 识别该帧中的分子
        molecules = identify_molecules_in_frame(atoms, mult_factor=mult_factor)
        
        # 遍历该帧中的每个分子
        for mol_idx, molecule in enumerate(molecules):
            formula = molecule['formula']
            
            # 如果这种分子类型尚未保存，则保存为PDB文件
            if formula not in saved_molecules and formula not in problem_molecules:
                # 获取该分子的原子索引
                atom_indices = molecule['atom_indices']
                
                if debug:
                    print(f"Processing molecule {formula} with {len(atom_indices)} atoms")
                
                # 提取分子并正确处理周期性边界条件
                molecule_atoms = extract_molecule_with_bonds(atoms, atom_indices, nl, debug=debug)
                
                if molecule_atoms is not None:
                    # 检查分子完整性
                    # is_valid, integrity_message = check_molecule_integrity(molecule_atoms)
                    # is_valid = True  # 默认认为分子是有效的
                    # integrity_message = "分子结构完整，没有检测到异常的原子距离或连通性问题。"

                    # 设置适当的文件名
                    filename = os.path.join(output_dir, f"{formula}.pdb")
                    
                    # 保存为PDB文件
                    io.write(filename, molecule_atoms)
                    
                    print(f"Saved molecule {formula} as {filename}")
                    
                    # 将该分子类型添加到已保存集合中
                    saved_molecules.add(formula)
                    saved_molecules_info[formula] = {
                        'filename': filename,
                        'num_atoms': len(molecule_atoms),
                        'composition': molecule['composition'],
                        'integrity': 'Valid'
                    }

    print(f"Total unique valid molecules saved: {len(saved_molecules)}")
    print(f"Total problematic molecules: {len(problem_molecules)}")
    
    # 合并有效分子和问题分子的信息
    all_molecules_info = {**saved_molecules_info, **problem_molecules}
    
    # 生成分子信息汇总表
    summary_df = pd.DataFrame.from_dict(all_molecules_info, orient='index')
    summary_file = os.path.join(output_dir, "molecules_summary.csv")
    summary_df.to_csv(summary_file)
    print(f"Saved molecules summary to {summary_file}")
    
    return list(saved_molecules)

def analyze_trajectory(trajectory_file: str, index=":", mult_factor=0.7) -> pd.DataFrame:
    """
    分析ASE轨迹文件中的每一帧，识别每一帧中的分子及其原子数量。
    返回一个包含每一帧分子信息的DataFrame。支持单帧和多帧文件。
    
    参数:
        trajectory_file: 轨迹文件路径
        index: 轨迹索引
        mult_factor: 截断半径系数，影响成键检测的灵敏度
    """
    # 读取轨迹文件
    try:
        # 先尝试读取为多帧轨迹
        traj = io.read(trajectory_file, index=index)
        
        # 检查是否为列表（多帧）或单个Atoms对象（单帧）
        if not isinstance(traj, list):
            traj = [traj]  # 将单帧转换为列表格式统一处理
    except Exception as e:
        print(f"Error reading trajectory: {e}")
        # 尝试作为单帧文件读取
        try:
            atoms = io.read(trajectory_file)
            traj = [atoms]
        except Exception as e2:
            print(f"Could not read file as a trajectory or a single frame: {e2}")
            return pd.DataFrame()

    # 用于存储每一帧的分子数据
    all_frames_data = []

    # 用于记录所有出现过的分子类型（列名）
    all_molecule_types = set()

    # 遍历轨迹的每一帧
    for frame_idx, atoms in enumerate(traj):
        frame_data = defaultdict(int)  # 用于存储该帧的分子计数
        print(f"Analyzing frame {frame_idx + 1} of {len(traj)}...")

        # 识别该帧中的分子
        molecules = identify_molecules_in_frame(atoms, mult_factor=mult_factor)
        molecule_names = [mol['formula'] for mol in molecules]

        # 统计每种分子出现的次数
        molecule_counts = Counter(molecule_names)

        # 更新所有出现过的分子类型
        all_molecule_types.update(molecule_counts.keys())

        # 将该帧的分子信息存入字典
        for molecule, count in molecule_counts.items():
            frame_data[molecule] = count
        
        # 将该帧的数据添加到所有帧的数据列表中
        all_frames_data.append(frame_data)

    # 将所有帧的数据转换为 DataFrame
    df = pd.DataFrame(all_frames_data)

    # 确保所有列都有统一的分子类型（如果某些分子在某些帧中不存在，填充为0）
    # 以 all_molecule_types 为列名创建列，缺失的分子数量用 0 填充
    df = df.reindex(columns=sorted(all_molecule_types), fill_value=0)

    # 设置DataFrame的行索引为帧号
    df.index = [f"{i + 1}" for i in range(len(df))]

    return df

# 定义一个主函数来运行整个流程
def main(trajectory_file, output_dir="unique_molecules", index=":", mult_factor=0.7, debug=False):
    """
    主函数：分析轨迹文件并保存唯一分子为PDB文件
    
    参数:
        trajectory_file: 轨迹文件路径
        output_dir: 输出目录
        index: 轨迹索引
        mult_factor: 截断半径系数，影响成键检测的灵敏度
        debug: 是否输出调试信息
    """
    print(f"Processing file: {trajectory_file}")
    print(f"Using cutoff multiplier: {mult_factor}")
    
    # 保存唯一分子为PDB文件
    unique_molecules = save_unique_molecules_as_pdb(
        trajectory_file, output_dir, index=index, mult_factor=mult_factor, debug=debug
    )
    print("Unique molecules:", unique_molecules)
    
    # 分析轨迹并创建DataFrame
    chem_df = analyze_trajectory(trajectory_file, index=index, mult_factor=mult_factor)
    
    if not chem_df.empty:
        chem_df = chem_df.fillna(0)
        print("Molecule counts per frame:")
        print(chem_df)
        
        # 可选：保存DataFrame到CSV文件
        csv_file = os.path.join(output_dir, "molecule_counts.csv")
        chem_df.to_csv(csv_file)
        print(f"Saved molecule counts to {csv_file}")
    else:
        print("No data to analyze or error in processing file.")
    
    return unique_molecules, chem_df

# 示例：调用主函数
if __name__ == "__main__":
    trajectory_file = "POSCAR"  # 替换为你的轨迹文件路径
    output_dir = "unique_molecules"  # 输出目录
    index = ":"  # 轨迹索引，对于单帧文件可以设置为":"
    
    # 增加截断半径系数，提高成键检测的敏感性
    mult_factor = 0.7  # 默认是0.7，增加到0.85可能会捕获更多的键
    
    # 启用调试输出
    debug = True
    
    # 对于单帧文件，可以使用
    # main(trajectory_file, output_dir, index=":", mult_factor=0.85, debug=True)
    
    # 对于多帧文件，可以使用指定索引
    main(trajectory_file, output_dir, index=index, mult_factor=mult_factor, debug=debug)
