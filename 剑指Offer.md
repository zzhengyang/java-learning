## 剑指Offer

[面试题03. 数组中重复的数字](#数组中重复的数字)

[面试题04. 二维数组中的查找](#二维数组中的查找)

[面试题05. 替换空格](#替换空格)

[面试题06. 从尾到头打印链表](#从尾到头打印链表)

[面试题07. 重建二叉树](#重建二叉树)

[面试题09. 用两个栈实现队列](#用两个栈实现队列)

[面试题10- I. 斐波那契数列](#斐波那契数列v)

[面试题10- II. 青蛙跳台阶问题](#青蛙跳台阶问题)

[面试题11. 旋转数组的最小数字](#旋转数组的最小数字)

[面试题12. 矩阵中的路径](#矩阵中的路径)


##  数组中重复的数字

>
>在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

>示例 1：
>
>输入：
>[2, 3, 1, 0, 2, 5, 3]
>输出：2 或 3 

```bash
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        """
        1. 先排序，再相邻元素比较
        时间复杂度O(nlogn),空间复杂度O(1)
        """
        nums.sort()
        pre = nums[0]
        for index in range(1, len(nums)):
            if nums[index] == pre:
                return pre
            pre = nums[index]

        """
        2. hashtable
        时间复杂度O(n), 空间复杂度O(n)
        """
        hashtab = {}
        for num in nums:
            if num not in hashtab:
                hashtab[num] = 1
            else:
                return num

        """
        3. 原地hash
        时间复杂度O(n), 空间复杂度O(1)
        """
        n = len(nums)
        for i in range(n):
            while i != nums[i]:
                if nums[i] == nums[nums[i]]:
                    return nums[i]
                temp = nums[i]
                nums[i], nums[temp] = nums[temp], nums[i]
                



```


## 二维数组中的查找

>在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。


>示例:
>
>现有矩阵 matrix 如下：
>
>[
>
>  [1,   4,  7, 11, 15],
> 
>  [2,   5,  8, 12, 19],
> 
>  [3,   6,  9, 16, 22],
> 
>  [10, 13, 14, 17, 24],
> 
>  [18, 21, 23, 26, 30]
> 
>]
>
>给定 target = 5，返回 true。
>
>给定 target = 20，返回 false。

```bash
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        """
        1. 遍历查找
        时间复杂度O(mn), 空间复杂度O(1)
        """
        if not matrix:
            return False
        rows, cols = len(matrix), len(matrix[0])
        for i in range(0, rows):
            for j in range(0, cols):
                print(matrix[i][j])
                if matrix[i][j] == target:
                    return True
        return False

        """
        2.从右上角比较，当前值比target大说明这一列往下都比target大，列减1；
        当前值比target小，说明这一行往左都比target小，行加1
        时间复杂度O(m+n), 空间复杂度O(1)
        """
        if not matrix:
            return False
        rows, cols = len(matrix), len(matrix[0])
        i, j = 0, cols - 1
        while i < rows and j >= 0:
            if matrix[i][j] < target:
                i += 1
            elif matrix[i][j] > target:
                j -= 1
            else:
                return True
        return False
```


## 替换空格

> 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

>示例 1：
>
>输入：s = "We are happy."
>输出："We%20are%20happy."

```bash
class Solution:
    def replaceSpace(self, s: str) -> str:
        """
        1.直接替换
        """
        return s.replace(' ', '%20')

        """
        2.创建新列表插入,再转换string
        """
        res = []
        for c in s:
            if c == ' ': res.append("%20")
            else: res.append(c)
        return "".join(res)
```

## 从尾到头打印链表


>输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

>示例 1：
>
>输入：head = [1,3,2]
>输出：[2,3,1]

```bash
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        """
        1. 递归法
        时间复杂度O(n), 空间复杂度O(n)
        """
        return self.reversePrint(head.next) + [head.val] if head else []

        """
        2. 辅助栈
        时间复杂度O(n), 空间复杂度O(n)
        """
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        return stack[::-1]

```

## 重建二叉树

>一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
>
>答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

>示例 1：
>
>输入：n = 2
>输出：2
>示例 2：
>
>输入：n = 7
>输出：21

```bash
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return None
        loc = inorder.index(preorder[0])
        root = TreeNode(preorder[0])
        root.left = self.buildTree(preorder[1 : loc + 1], inorder[: loc])
        root.right = self.buildTree(preorder[loc + 1: ], inorder[loc + 1: ])
        return root
```


## 用两个栈实现队列

>用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

 

>示例 1：
>
>输入：

>["CQueue","appendTail","deleteHead","deleteHead"]
>
>[[],[3],[],[]]
>
>输出：[null,null,3,-1]

>示例 2：
>
>输入：
>
>["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
>
>[[],[],[5],[2],[],[]]
>
>输出：[null,-1,null,null,5,2]

```bash
class CQueue:
    """
    插入元素
    时间复杂度：O(n)O(n)。插入元素时，对于已有元素，每个元素都要弹出栈两次，压入栈两次，因此是线性时间复杂度。
    空间复杂度：O(n)O(n)。需要使用额外的空间存储已有元素。

    删除元素
    时间复杂度：O(1)O(1)。判断元素个数和删除队列头部元素都使用常数时间。
    空间复杂度：O(1)O(1)。从第一个栈弹出一个元素，使用常数空间。

    """
    def __init__(self):
        self.stackin = []
        self.stackout = []

    def appendTail(self, value: int) -> None:
        self.stackin.append(value)

    def deleteHead(self) -> int:
        if not self.stackout:
            if not self.stackin:
                return -1
            while self.stackin:
                self.stackout.append(self.stackin.pop())
        return self.stackout.pop()



# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```

## 斐波那契数列

>写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：
>
>F(0) = 0,   F(1) = 1
>
>F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
>
>斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。
>
>答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

 

>示例 1：
>
>输入：n = 2
>
>输出：1

>示例 2：
>
>输入：n = 5
>输出：5

```bash
class Solution:
    @lru_cache(None)
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for i in range(n):
            a, b = b, a + b
        return a % 1000000007
```


## 青蛙跳台阶问题

>一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
>
>答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

>示例 1：
>
>输入：n = 2
>
>输出：2

>示例 2：
>
>输入：n = 7
>
>输出：21

```bash
class Solution:
    def numWays(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007
```

## 旋转数组的最小数字

>把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，>输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  
>
>示例 1：
>
>输入：[3,4,5,1,2]
>输出：1
>示例 2：
>
>输入：[2,2,2,0,1]
>输出：0

```bash
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left, right = 0, len(numbers) - 1
        while left < right:
            mid = (left + right) // 2
            if numbers[mid] > numbers[right]: left = mid + 1
            elif numbers[mid] < numbers[right]: right = mid
            else: right -= 1
        return numbers[left]
```
**复杂度分析：**

时间复杂度 $O(log_2N)$： 在特例情况下（例如 [1, 1, 1, 1]），会退化到 $O(N)$。

空间复杂度 $O(1)$ ： i , j , m 指针使用常数大小的额外空间。


## 矩阵中的路径

>请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。
>
>[["a","b","c","e"],
>
>["s","f","c","s"],
>
>["a","d","e","e"]]
>
>但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

>示例 1：
>
>
>输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
>输出：true

>示例 2：
>
>输入：board = [["a","b"],["c","d"]], word = "abcd"
>
>输出：false

>**提示**：
>
>* 1 <= board.length <= 200
>
>* 1 <= board[i].length <= 200

```bash
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, k):
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]: return False
            if k == len(word) - 1: return True
            tmp, board[i][j] = board[i][j], '/'
            res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
            board[i][j] = tmp
            return res
        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0): return True
        return False
```

**复杂度分析：**

>M, N 分别为矩阵行列大小， K 为字符串 word 长度。

时间复杂度 $O(3^KMN)$: 最差情况下，需要遍历矩阵中长度为 $K$ 字符串的所有方案，时间复杂度为 $O(3^K)$；矩阵中共有 $MN$ 个起点，时间复杂度为 $O(MN)$。 

方案数计算： 设字符串长度为 $K$ ，搜索中每个字符有上、下、左、右四个方向可以选择，舍弃回头（上个字符）的方向，剩下 3 种选择，因此方案数的复杂度为 $O(3^K)$ 。

空间复杂度 $O(K)$ ： 搜索过程中的递归深度不超过 $K$ ，因此系统因函数调用累计使用的栈空间占用 $O(K)$（因为函数返回后，系统调用的栈空间会释放）。最坏情况下 $K = MN$ ，递归深度为 $MN$ ，此时系统栈使用 $O(MN)$ 的额外空间。






















