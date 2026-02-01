\section{Project Description}

\subsection{Aim of project}
\begin{itemize}
    \item Explore an interesting machine learning problem
    \item Gain machine learning development experience
    \item See applications of theory covered in this course
    \item Read published machine learning papers
\end{itemize}

\section{Project Requirements}

\subsection{Paper selection constraints}
\begin{itemize}
    \item Select a published paper on machine learning
    \item Must be published in 2018 or later
    \item Must be from a Q1 journal or a top-tier conference
    \item Examples of Q1 journals (list not exhaustive):
    \begin{itemize}
        \item JMLR
        \item TPAMI
        \item IJCV
    \end{itemize}
    \item Acceptable conferences include, but are not limited to:
    \begin{itemize}
        \item General ML/AI: NeurIPS, ICML, ICLR, AAAI, IJCAI
        \item Computer Vision: CVPR, ICCV, ECCV
    \end{itemize}
    \item You can also use CORE A or CORE A* venues
    \item You can check the ranking of a venue using:
\end{itemize}
\noindent\texttt{https://portal.core.edu.au/conf-ranks/}
\begin{itemize}
    \item arXiv does not count
\end{itemize}

\subsection{Implementation constraints}
\begin{itemize}
    \item Implement the model in the paper with the following requirements:
\end{itemize}

\subsubsection{Domain Shift}
\begin{itemize}
    \item The model must be applied to a completely different domain
    \item Example: if the paper is about natural image segmentation, apply it to medical imaging or satellite data
\end{itemize}

\subsubsection{Performance Audit}
\begin{itemize}
    \item Final submission must not just show that the model works
    \item You must present a sensitivity analysis or an ablation study showing what happens when the model breaks
\end{itemize}

\section{Project Milestones}

\subsection{Initial project proposal}
\begin{itemize}
    \item Purpose: provide immediate feedback on the suitability and feasibility of your proposed paper + dataset pairing
    \item Students will pitch their idea early in the course to ensure:
    \begin{itemize}
        \item the concept meets the difficulty requirements, and
        \item the new dataset is compatible with the chosen architecture
    \end{itemize}
    \item Presentation consists of one slide and is 3 minutes in duration
    \item You must clearly identify:
    \begin{itemize}
        \item The specific paper (Title + Venue)
        \item A general description of the model
        \item The new dataset you intend to use
        \item The core technical challenge in adapting the former to the latter
    \end{itemize}
    \item Presentation date: January 22, 2026
    \item Submission: slides must be submitted to UM Learn by 5:00 pm on January 21, 2026
\end{itemize}

\subsection{Full Project Proposal}
\begin{itemize}
    \item Students will present a detailed technical proposal for their project via a slide presentation
    \item This is a ``gate'' presentation (used to verify the group has planned the project carefully)
    \item Logistics:
    \begin{itemize}
        \item Duration: 5--10 minutes
        \item You will be stopped promptly at the 10-minute mark
        \item Q\&A: each student in the group may be asked 1--2 individual questions to verify their specific contribution and understanding
    \end{itemize}
    \item Dates: February 3 and February 5, 2026
    \item Groups will be randomly selected at the start of class
    \item Submission: slides must be submitted to UM Learn by 5:00 pm on February 2, 2026
\end{itemize}

\section{Project Milestones and Deliverables}

\subsection{Full Project Proposal: required slide content}
\begin{itemize}
    \item Proposal must go beyond the initial pitch and demonstrate technical feasibility
    \item Slides should include:
\end{itemize}

\subsubsection{Title Slide}
\begin{itemize}
    \item Project name
    \item Group member names
    \item Course name
    \item Instructor's name
    \item Date
\end{itemize}

\subsubsection{Problem Domain \& Dataset}
\begin{itemize}
    \item Introduce the specific problem you are solving
    \item Dataset: detailed description of your chosen dataset, including:
    \begin{itemize}
        \item size
        \item resolution
        \item file format
        \item class distribution
    \end{itemize}
    \item You must explicitly confirm this dataset is distinct from the dataset used in the original paper
\end{itemize}

\subsubsection{Source Paper \& Methodology}
\begin{itemize}
    \item Identify the specific paper (Title + Venue, 2018+)
    \item Architecture diagram: a high-level diagram of the model architecture you will build
\end{itemize}

\subsubsection{Adaptation Strategy}
\begin{itemize}
    \item Explain the specific technical changes required to apply the paper's architecture to your new dataset
    \item Examples:
    \begin{itemize}
        \item Change input channels from 3 (RGB) to 5 (multispectral)
        \item Rewrite the data loader to handle 3D volumes instead of 2D images
    \end{itemize}
\end{itemize}

\subsubsection{Development Plan}
\begin{itemize}
    \item Environment: state the framework (e.g., PyTorch) and hardware resources (e.g., Dept GPUs, Colab) you will use
    \item Ablation plan: identify at least one ablation study you plan to perform
    \begin{itemize}
        \item Example: remove a specific module or change a hyperparameter to test robustness
    \end{itemize}
\end{itemize}

\subsubsection{Project Management}
\begin{itemize}
    \item Gantt chart: a timeline with clear milestones
    \item Example milestones:
    \begin{itemize}
        \item Data pipeline working
        \item Baseline training
        \item Final analysis
    \end{itemize}
\end{itemize}

\subsubsection{Role Division}
\begin{itemize}
    \item Clearly state which student is responsible for which technical component
\end{itemize}

\subsubsection{EDI Considerations}
\begin{itemize}
    \item Discuss Equity, Diversity, and Inclusion considerations relevant to your dataset or problem
    \item Examples:
    \begin{itemize}
        \item bias in the training data
        \item accessibility
    \end{itemize}
    \item Resource:
\end{itemize}
\noindent\texttt{https://nserc-crsng.canada.ca/en/nserc-guide-integrating-equity-diversity-and-inclusion-considerations-research}

\subsubsection{References}
\begin{itemize}
    \item Proper citations for the source paper and dataset (and any other sources referenced in the presentation)
    \item Formatted using IEEE citation style
\end{itemize}
