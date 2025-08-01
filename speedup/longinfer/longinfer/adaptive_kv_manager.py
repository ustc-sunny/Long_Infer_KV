import torch
import torch.nn.functional as F

#from kv_abstract import get_kv_abstract
import heapq

# def get_min_max_score(query, kv_abstract):
#     max_scores = torch.sum(torch.max(query * kv_abstract[0], query * kv_abstract[1]))
#     min_scores = torch.sum(torch.min(query * kv_abstract[0], query * kv_abstract[1]))
#     return min_scores, max_scores
def get_kv_abstract(kv_chunk):
    min_vector, _ = torch.min(kv_chunk, dim=0)
    max_vector, _ = torch.max(kv_chunk, dim=0)
    return min_vector, max_vector

def get_min_max_score(query, kv_abstract):
    """根据当前查询和键值摘要计算最小和最大分数。"""
    min_vector, max_vector = kv_abstract
    # 计算每个注意力头的分数
    max_scores = torch.sum(torch.max(query * min_vector, query * max_vector), dim=-1)
    min_scores = torch.sum(torch.min(query * min_vector, query * max_vector), dim=-1)
    return min_scores, max_scores

def evaluetion(k, query):
     return torch.sum(k * query, dim=-1)

#选择重要的token id
#这段代码主要是通过curent_q，all_k的数据进行近似attention计算以评估重要性，然后选择前importance_ratio比例的
#token的KV数据的ID，也就是prefetch_idx。以上是最主要的功能，再就是返回是重要的KV abstract中的kv_abstract_idx。
# 具体流程如下：首先对all_k进行chunk划分，大小为chunk_size，然后再对其进行 KV absrtact操作得到每个chunk的
# min_vector和max_vector。然后将其与传进来的kv_abstract先后计算得到最大得分和最小得分，其中注意区分两者，
# 注意区分index。将其存入一个优先队列，这个优先队列始终是按照chunk的最大得分降序排序。然后根据最大得分选择前
# importance_ratio比例token的chunks，（取上界，也就是chunks中token数量加起来略大于importance_ratio比例）。
# 然后根据最大值得分判断可以出现在前important_ratio比例的kv abstract的index保存下来（也就是大于前
# importance_ratio比例中最小的最大得分的kv abstract的index），作为返回值kv_abstract_idx。然后开始迭代，
# 迭代次数自己决定。一个迭代中是这样的。将选中的前importance_ratio比例的chunks一分为二，然后计算其最大得分
# 与最小得分。然后继续根据最大得分选择前importance_ratio比例token的chunks（因为一分为二了，后边的chunk可能
# 排进了前inportance_ratio）。然后在选出来的chunk中，把没有一分为二的（原来没有选择出来的）chunk一分为二，
# 然后再次选择前importance_ratio。主要要记录选择出来的chunk的对应token的index，最后作为返回prefetch_idx返回

def IAKM_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv, current_q, all_k, importance_ratio, chunk_size, kv_abstract):
    """选择重要的 token ID。"""
    # Step 1: Split all_k into chunks of size chunk_size
    # 调整 all_k 的形状从 [384, 32, 64] 到 [384, 32*64]
    #reshape = all_k.shape[2]
    IAKM = 2# 1:longinfer 0:H2O-token 2:chunked
    if IAKM == 2:
        all_k = all_k.reshape(all_k.shape[0], -1)  # [384, 32*64]
        # 调整 query 的形状从 [32, 1, 64] 到 [1, 32*64]
        current_q = current_q.reshape(1, -1)  # [1, 32*64]
        #print("all_k.shape=",all_k.shape)
        #print("current_q.shape=",current_q.shape)
        chunks = torch.split(all_k, chunk_size, dim=0)
        #print("chunks_len=",len(chunks),len(chunks[0]),len(chunks[-1]))
        
        # Step 2: Evaluate the importance of each chunk by calculating min and max vectors
        importance_scores = []
        for idx, chunk in enumerate(chunks):
            min_vector, max_vector = get_kv_abstract(chunk)
            min_scores, max_scores = get_min_max_score(current_q, (min_vector, max_vector))
            importance_scores.append((idx, max_scores.item(), min_scores.item(),[idx*chunk_size,(idx+1)*chunk_size-1]))  # 存储索引、min_scores 和 max_scores, token index范围

        # Step 3: Sort the chunks by max score in descending order
        priority_queue = sorted(importance_scores, key=lambda x: x[1], reverse=True)
        #print("priority_queue=",priority_queue)
        # Step 4: Determine the total number of tokens to select based on importance_ratio
        total_tokens = int(all_k.shape[0] * importance_ratio)
        n_selected_chunks = int(len(chunks)*importance_ratio)
        
        selected_chunks = priority_queue[:n_selected_chunks*2]  # Select top n_selected_chunks*2 based on max score
        min_selected_score = selected_chunks[-1][1] if selected_chunks else 0

        # Step 7: Select KV abstracts from the parameter kv_abstract that have a higher score than min_selected_score
        if kv_abstract is not None:
            kv_abstract_scores = []
            for abstract_idx, (min_vec, max_vec) in enumerate(kv_abstract):
                _, max_score = get_min_max_score(current_q, (min_vec, max_vec))
                kv_abstract_scores.append((max_score.item(), abstract_idx))
            # Sort KV abstracts by score and select those above min_selected_score
            #kv_abstract_scores.sort(key=lambda x: x[0], reverse=True)
            kv_abstract_idx = []
            for score, idx in kv_abstract_scores:
                if score >= min_selected_score:
                    kv_abstract_idx.append(idx)
            kv_abstract_idx = torch.tensor(kv_abstract_idx)
        else:
            kv_abstract_idx = None

        prefetch_idx = []
        if chunk_size == 8:
            for chn in selected_chunks:
                for i in range(chn[3][0],chn[3][1]+1):
                    prefetch_idx.append(i)
            #prefetch_idx = torch.tensor(prefetch_idx)
            #print("8:prefetch_idx=",prefetch_idx)

        # Step 8: Iterative refinement process
        #print("selected_chunks=",selected_chunks)
        # Step 5: Calculate the importance scores for each token in the selected chunks
        importance_scores = []
        for chn in selected_chunks:
            for i in range(chn[3][0],chn[3][1]):
                if i < all_k.shape[0]:
                    importance_scores.append((i, evaluetion(all_k[i], current_q)))
        importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)
        #print("importance_scores=",importance_scores)
        prefetch_idx = []
        for i in importance_scores[:total_tokens]:
            prefetch_idx.append(i[0])
        prefetch_idx = torch.tensor(prefetch_idx)
    elif IAKM == 0:
        all_k = all_k.reshape(all_k.shape[0], -1)  # [384, 32*64]
        # 调整 query 的形状从 [32, 1, 64] 到 [1, 32*64]
        current_q = current_q.reshape(1, -1)  # [1, 32*64]
        #print("all_k.shape=",all_k.shape)
        #print("current_q.shape=",current_q.shape)
        chunks = torch.split(all_k, chunk_size, dim=0)
        #print("chunks_len=",len(chunks),len(chunks[0]),len(chunks[-1]))
        
        # Step 2: Evaluate the importance of each chunk by calculating min and max vectors
        importance_scores = []
        for idx, chunk in enumerate(chunks):
            min_vector, max_vector = get_kv_abstract(chunk)
            min_scores, max_scores = get_min_max_score(current_q, (min_vector, max_vector))
            importance_scores.append((idx, max_scores.item(), min_scores.item(),[idx*chunk_size,(idx+1)*chunk_size-1]))  # 存储索引、min_scores 和 max_scores, token index范围

        # Step 3: Sort the chunks by max score in descending order
        priority_queue = sorted(importance_scores, key=lambda x: x[1], reverse=True)
        #print("priority_queue=",priority_queue)
        # Step 4: Determine the total number of tokens to select based on importance_ratio
        total_tokens = int(all_k.shape[0] * importance_ratio)
        n_selected_chunks = int(len(chunks)*importance_ratio)
        
        selected_chunks = priority_queue[:n_selected_chunks*2]  # Select top n_selected_chunks*2 based on max score
        min_selected_score = selected_chunks[-1][1] if selected_chunks else 0

        # Step 7: Select KV abstracts from the parameter kv_abstract that have a higher score than min_selected_score
        if kv_abstract is not None:
            kv_abstract_scores = []
            for abstract_idx, (min_vec, max_vec) in enumerate(kv_abstract):
                _, max_score = get_min_max_score(current_q, (min_vec, max_vec))
                kv_abstract_scores.append((max_score.item(), abstract_idx))
            # Sort KV abstracts by score and select those above min_selected_score
            #kv_abstract_scores.sort(key=lambda x: x[0], reverse=True)
            kv_abstract_idx = []
            for score, idx in kv_abstract_scores:
                if score >= min_selected_score:
                    kv_abstract_idx.append(idx)
            kv_abstract_idx = torch.tensor(kv_abstract_idx)
        else:
            kv_abstract_idx = None

        prefetch_idx = []
        if chunk_size == 8:
            for chn in priority_queue:
                for i in range(chn[3][0],chn[3][1]+1):
                    prefetch_idx.append(i)
            #prefetch_idx = torch.tensor(prefetch_idx)
            #print("8:prefetch_idx=",prefetch_idx)

        # Step 8: Iterative refinement process
        #print("selected_chunks=",selected_chunks)
        # Step 5: Calculate the importance scores for each token in the selected chunks
        importance_scores = []
        for chn in priority_queue:
            for i in range(chn[3][0],chn[3][1]):
                if i < all_k.shape[0]:
                    importance_scores.append((i, evaluetion(all_k[i], current_q)))
        importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)
        #print("importance_scores=",importance_scores)
        prefetch_idx = []
        for i in importance_scores[:total_tokens]:
            prefetch_idx.append(i[0])
        prefetch_idx = torch.tensor(prefetch_idx)
    else:
        all_k = all_k.reshape(all_k.shape[0], -1)  # [384, 32*64]
        # 调整 query 的形状从 [32, 1, 64] 到 [1, 32*64]
        current_q = current_q.reshape(1, -1)  # [1, 32*64]
        #print("all_k.shape=",all_k.shape)
        #print("current_q.shape=",current_q.shape)
        chunks = torch.split(all_k, chunk_size, dim=0)
        #print("chunks_len=",len(chunks),len(chunks[0]),len(chunks[-1]))
        
        # Step 2: Evaluate the importance of each chunk by calculating min and max vectors
        importance_scores = []
        for idx, chunk in enumerate(chunks):
            min_vector, max_vector = get_kv_abstract(chunk)
            min_scores, max_scores = get_min_max_score(current_q, (min_vector, max_vector))
            importance_scores.append((idx, max_scores.item(), min_scores.item(),[idx*chunk_size,(idx+1)*chunk_size-1]))  # 存储索引、min_scores 和 max_scores, token index范围

        # Step 3: Sort the chunks by max score in descending order
        priority_queue = sorted(importance_scores, key=lambda x: x[1], reverse=True)
        #print("priority_queue=",priority_queue)
        # Step 4: Determine the total number of tokens to select based on importance_ratio
        total_tokens = int(all_k.shape[0] * importance_ratio)
        n_selected_chunks = int(len(chunks)*importance_ratio)
        
        selected_chunks = priority_queue[:n_selected_chunks*2]  # Select top n_selected_chunks*2 based on max score
        min_selected_score = selected_chunks[-1][1] if selected_chunks else 0

        # Step 7: Select KV abstracts from the parameter kv_abstract that have a higher score than min_selected_score
        if kv_abstract is not None:
            kv_abstract_scores = []
            for abstract_idx, (min_vec, max_vec) in enumerate(kv_abstract):
                _, max_score = get_min_max_score(current_q, (min_vec, max_vec))
                kv_abstract_scores.append((max_score.item(), abstract_idx))
            # Sort KV abstracts by score and select those above min_selected_score
            #kv_abstract_scores.sort(key=lambda x: x[0], reverse=True)
            kv_abstract_idx = []
            for score, idx in kv_abstract_scores:
                if score >= min_selected_score:
                    kv_abstract_idx.append(idx)
            kv_abstract_idx = torch.tensor(kv_abstract_idx)
        else:
            kv_abstract_idx = None

        prefetch_idx = []
        if True:
            for chn in priority_queue:
                for i in range(chn[3][0],chn[3][1]+1):
                    prefetch_idx.append(i)

    b = hidden.shape[0]
    p_q = F.linear(hidden, p_w_q, bias=None)
    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
    max_ = torch.max(p_attn, dim=-1)[0]
    thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    count = torch.where(
        p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    )
    mean = torch.mean(torch.sum(count, dim=-1)).item()
    prefetch_idx = torch.topk(
        p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
    )[1]

    return prefetch_idx, kv_abstract_idx
    #return prefetch_idx, kv_abstract_idx

def select_kv(prefetch_idx, k_cache, v_cache):
    # 确保 prefetch_idx 在正确的设备上
    prefetch_idx = prefetch_idx.to(k_cache.device)
    
    # 根据 prefetch_idx 选择对应的键和值缓存
    selected_k = k_cache[prefetch_idx]
    selected_v = v_cache[prefetch_idx]
    #print("selected_k.shape=",selected_k.shape)
    return selected_k, selected_v
    
def _select_kv(prefetch_idx, k_cache, v_cache):
    """Selects and aggregates critical KV caches using speculated indices

    On the decoding stage, aggregates the critical KV caches corresponding to
    the speculated prefetch index using embedding function.

    Args:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)

    Returns:
        selected_k: selected key cache (n', bh, d)
        selected_v: selected value cache (n', bh, d)
    """

    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v

def set_partial_cache(k_cache, partial_index, n_head, head_dim):
    """Sets the partial key cache.

    On the prefill and decoding stages, generates the partial key cache
    following the partial_index which indicates the indices of the important
    columns.

    Args:
        k_cahce: Key cache (n, bh, d)
        partial_weight_index: Indices of top-k columns (b, h, d')
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_cache: Partial key cache (n, bh, d')
    """

    n, bh, _ = k_cache.shape
    partial_cache = torch.gather(
        k_cache.view(n, -1, n_head, head_dim),
        3,
        partial_index.unsqueeze(0).repeat(n, 1, 1, 1),
    )
    return partial_cache.view(n, bh, -1)

def reform_hidden_states(hidden_states):
    """Concatenates the weight matrix and bias.

    Concatenates the hidden states with a column of 1.
    This reformation with the concatenated weight and bias  makes the linear
    projection into a one matrix multiplication without bias addition.

    Args:
        hidden: Hidden states (b, n, D)

    Returns:
        reformed hidden states (b, n, D+1)
    """

    return torch.cat(
        (hidden_states, torch.ones_like(hidden_states)[:, :, 1].unsqueeze(2)), dim=-1
    )

def weight_bias_concat(weight, bias, scaling=False, head_dim=1.0):#warmup阶段需要的
    """Concatenates the weight matrix and bias.

    On the warmup phase, concatenates the weight matrix and bias for skewing.
    This manipulation does not hurt the correctness.

    Args:
        weight: Weight matrix (D, D)
        bias: Bias vector (D)
        scaling: If ture, scales the concatenated weight and bias to skip
            the scaling after projection.
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        concatenated weight and bias (D, D+1)
    """

    if scaling:
        return torch.cat((weight, bias.unsqueeze(1).to(weight.device)), dim=1) * (
            head_dim**-0.5
        )
    else:
        return torch.cat((weight, bias.unsqueeze(1).to(weight.device)), dim=1)
    
def set_partial_weight(w_q, partial_index, n_head, head_dim):
    """Sets the partial query weight.

    On the prefill stage, generates the partial query weight following the
    partial_index which indicates the indices of the important columns.

    Args:
        w_q: Query weight (D, D)
        partial_weight_index: Indices of top-k columns (b, h, d')
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        partial_weight: Partial query weight (D', D)
    """

    partial_weight = F.embedding(
        partial_index[0]
        + torch.arange(n_head)[:, None].to(partial_index.device) * head_dim,
        w_q.view(-1, w_q.shape[-1]),
    )
    return partial_weight.view(-1, w_q.shape[-1])
'''
def IAKM_attention(current_q, all_k, importance_ratio, max_num_kv, chunk_size, kv_abstract):
    """Speculates the indices of the critical KV caches of next attention layer.
    Returns:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
    """
    iterations = 3
    # Step 1: Split all_k into chunks of size chunk_size
    chunks = torch.split(all_k, chunk_size, dim=0)

    # Step 2: Evaluate the importance of each chunk
    importance_scores = []
    for chunk in chunks:
        min_vector, max_vector = get_kv_abstract(chunk)
        min_scores, max_scores = get_min_max_score(current_q, [min_vector, max_vector])
        importance_scores.append((min_scores, max_scores))

    # Step 3: Calculate the upper and lower bounds for each chunk
    bounds = [(min_score.item(), max_score.item()) for min_score, max_score in importance_scores]

    # Step 4: Initialize a priority queue with chunks sorted by upper bound
    priority_queue = [(-bounds[i][1], i) for i in range(len(bounds))]
    heapq.heapify(priority_queue)

    # Determine the total number of tokens to select
    total_tokens = int(all_k.shape[0] * importance_ratio)
    selected_chunks = set()

    # Iterative refinement process
    for _ in range(iterations):  # `iterations` is a parameter to control the number of refinements
        # Select chunks to meet the importance ratio
        selected_indices = []
        current_token_count = 0

        while priority_queue and current_token_count < total_tokens:
            _, idx = heapq.heappop(priority_queue)
            selected_indices.append(idx)
            current_token_count += min(chunk_size, all_k.shape[0] - idx * chunk_size)

        # Split selected chunks into smaller chunks
        refined_chunks = []
        for idx in selected_indices:
            chunk_start = idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, all_k.shape[0])
            chunk = all_k[chunk_start:chunk_end]

            # Split the chunk into two halves
            mid_point = (chunk_end - chunk_start) // 2
            refined_chunks.append((chunk[:mid_point], idx))
            refined_chunks.append((chunk[mid_point:], idx))

        # Recalculate bounds for refined chunks
        new_bounds = []
        for chunk, original_idx in refined_chunks:
            min_vector, max_vector = get_kv_abstract(chunk)
            min_scores, max_scores = get_min_max_score(current_q, [min_vector, max_vector])
            new_bounds.append((-max_scores.item(), original_idx))

        # Update the priority queue with refined chunks
        priority_queue = new_bounds
        heapq.heapify(priority_queue)

        # Update the set of selected chunks
        selected_chunks.update(selected_indices)

    # Convert selected chunks to token indices
    prefetch_idx = []
    for idx in selected_chunks:
        chunk_start = idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, all_k.shape[0])
        prefetch_idx.extend(range(chunk_start, chunk_end))

    # Convert prefetch_idx to a tensor
    prefetch_idx = torch.tensor(prefetch_idx).unsqueeze(1)

    return prefetch_idx, kv_abstract_idx
'''