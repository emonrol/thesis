\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{editorialupv}[2016/09/05 Editorial de la Universitat Polit?cnica de Val?ncia]
\LoadClass[10pt]{book}

% ------------------------------------------------------

\RequirePackage{ifthen}

% ------------------------------------------------------

\newboolean{LLIBRE}\setboolean{LLIBRE}{false}
\newboolean{TESI}\setboolean{TESI}{false}
\newboolean{Aquatre}\setboolean{Aquatre}{false}
\newboolean{EBOOKPDF}\setboolean{EBOOKPDF}{false}
\newboolean{EPUB}\setboolean{EPUB}{false}

\newboolean{CROP}\setboolean{CROP}{false}

\newboolean{DistanciesEquacions}\setboolean{DistanciesEquacions}{true}

\newboolean{RM}\setboolean{RM}{false}
\newboolean{SF}\setboolean{SF}{false}

\newboolean{valencia}\setboolean{valencia}{false}
\newboolean{castellano}\setboolean{castellano}{false}
\newboolean{english}\setboolean{english}{false}


% ------------------------------------------------------

\DeclareOption{EPUB}{\setboolean{EPUB}{true}}
\DeclareOption{llibre}{\setboolean{LLIBRE}{true}}
\DeclareOption{tesi}{\setboolean{TESI}{true}}
\DeclareOption{a4}{\setboolean{Aquatre}{true}}
\DeclareOption{ebookpdf}{\setboolean{EBOOKPDF}{true}}

\DeclareOption{no-crop}{\setboolean{CROP}{false}}
\DeclareOption{crop}{\setboolean{CROP}{true}}
% Per compatibilitat
\DeclareOption{nocrop}{\setboolean{CROP}{false}}
\DeclareOption{croscrop}{\setboolean{CROP}{true}}

\DeclareOption{rm}{\setboolean{RM}{true}}
\DeclareOption{sf}{\setboolean{SF}{true}}

\DeclareOption{nomathskip}{\setboolean{DistanciesEquacions}{false}}

\DeclareOption{castellano}{
	\setboolean{castellano}{true}
	\setboolean{valencia}{false}
	\setboolean{english}{false}
}
\DeclareOption{valencia}{
	\setboolean{castellano}{false}
	\setboolean{valencia}{true}
	\setboolean{english}{false}
	}
\DeclareOption{english}{
	\setboolean{castellano}{false}
	\setboolean{valencia}{false}
	\setboolean{english}{true}
	}

% ------------------------------------------------------

\DeclareOption*{\ClassWarning{editorialupv}{No podem processar l'opci\'{o}: ?\CurrentOption?}}

% ------------------------------------------------------

\ExecuteOptions{llibre,rm,nocrop,castellano}
\ProcessOptions\relax


% ------------------------------------------------------
% Comprovem si s'està executant htlatex

\ifdefined\HCode % S'ha executat htlatex
	\setboolean{EPUB}{true}
	\setboolean{LLIBRE}{false}
	\setboolean{Aquatre}{false}
	\setboolean{TESI}{false}
	\setboolean{EBOOKPDF}{false}
\else % S'ha executat pdflatex o latex
	\ifEPUB % Es canvia a llibre si està seleccionat EPUB
		\setboolean{EPUB}{false}
		\setboolean{LLIBRE}{true}
\fi

% ------------------------------------------------------

\RequirePackage{graphicx}
\RequirePackage{calc}
%\RequirePackage{hyperref}

% ------------------------------------------------------

\ifthenelse{\boolean{LLIBRE}}{%
  \renewcommand\maketitle{
  \begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \begin{center}%
  \vskip 400\p@
  {EDITORIAL\\UNIVERSITAT POLIT\`ECNICA DE VAL\`ENCIA}
  \vskip -500\p@
	\rule{\textwidth}{1pt}\par
    \vskip 1.5em%
    {\Huge\bfseries \@title \par}%
    \vskip 1.5em%
	\rule{\textwidth}{1pt}\par
    {\large
     \lineskip .75em%
	  \vskip 60\p@
      \begin{tabular}[t]{c}%
        \@author\par
      \end{tabular}\par}%
	  \vskip 60\p@
%    {\large \@date \par}%       % Set date in \large size.
%  \@thanks
  \vfil\null
  \end{center}\par
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}}{}

\ifthenelse{\boolean{TESI}}{%
  \renewcommand\maketitle{
  \begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \begin{center}%
  \vskip -50\p@
  {\includegraphics[width=4.0cm]{./logos/UPV_horitzontal}}
  \vskip 10\p@
  DEPARTAMENTO DE MATEMÁTICAS
  \vskip 40\p@
	\rule{\textwidth}{1pt}\par
    \vskip 1.5em%
    {\Huge\bfseries \@title \par}%
    \vskip 1.5em%
	\rule{\textwidth}{1pt}\par
    {\large
     \lineskip .75em%
	  \vskip 60\p@
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
	  \vskip 60\p@
%    {\large \@date \par}%       % Set date in \large size.
%  \@thanks
  \vfil\null
  \end{center}\par
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}}{}

\ifthenelse{\boolean{EBOOKPDF}}{%
  \renewcommand\maketitle{
%  \begin{titlepage}% Si la comentem compta com a p?gina del PDF
  \thispagestyle{plain}
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \begin{center}%
  \vskip 355\p@
  \includegraphics[width=4.0cm]{./logos/UPV_horitzontal}\\[-.25cm]
  \rule{4cm}{0.25pt}\\
  {\sffamily\scriptsize\bfseries EDITORIAL}
  \vskip -415\p@
	\rule{\textwidth}{1pt}\par
    \vskip 1.5em%
    {\Huge\bfseries \@title \par}%
    \vskip 1.5em%
	\rule{\textwidth}{1pt}\par
    {\large
     \lineskip .75em%
	  \vskip 40\p@
      \begin{tabular}[t]{c}%
        \@author\par
      \end{tabular}\par}%
	  \vskip 60\p@
%    {\large \@date \par}%       % Set date in \large size.
%  \@thanks
  \vfil\null
  \end{center}\par
%  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  \clearpage
}}{}

\ifthenelse{\boolean{Aquatre}}{%
  \renewcommand\maketitle{
  \begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \begin{center}%
	\vskip 0\p@
	\includegraphics[width=5.0cm]{./logos/UPV_horitzontal}\\[-.25cm]
	\vskip 200\p@
    {\Huge\bfseries \@title \par}%
    {\large
     \lineskip .75em%
	  \vskip 150\p@
      \begin{tabular}[t]{c}%
        \@author\par
      \end{tabular}\par}%
%    {\large \@date \par}%
  \vfil\null
  \end{center}\par
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}}{}


% ---------------------------------------------------------------------
% Tipus de lletra

\ifLLIBRE
	\renewcommand{\normalsize}{\fontsize{10.5pt}{12.5pt}\selectfont}
\fi

\ifTESI
%	\renewcommand{\normalsize}{\fontsize{10.5pt}{12.5pt}\selectfont}
    \renewcommand{\normalsize}{\fontsize{9.5pt}{11.5pt}\selectfont}
\fi

\ifAquatre
	\renewcommand{\normalsize}{\fontsize{11pt}{13pt}\selectfont}
\fi

\ifEBOOKPDF
	\renewcommand{\familydefault}{\rmdefault}
	\renewcommand{\normalsize}{\fontsize{12pt}{14pt}\selectfont}
\fi

% ---------------------------------------------------------------------

\ifSF
	\renewcommand{\familydefault}{\sfdefault}
\else
	\renewcommand{\familydefault}{\rmdefault}
\fi
	

% ---------------------------------------------------------------------
% Format de pàgina

\RequirePackage{geometry}

\ifEBOOKPDF
	
	\geometry{
		twoside = false,
		body={4.75in,6.6in},
		hmarginratio=1:1,
		includeheadfoot,
%		paperwidth = 5.5in,
%		paperheight = 7.5in,
		paperheight=7.60606060in,
   		paperwidth=5.8181818in,
   		vmargin={0cm},
   		headsep = 1.25cm,	
		}

\else\ifAquatre
	
	\geometry{
		a4paper, twoside,         	% A4 a doble cara
		hmargin = {3.0cm, 2.25cm}, 	% {Izquierdo, Derecho}
		vmargin = {1.25cm, 1.25cm}, % {Superior, Inferior}
		includehead, includefoot, 	% Incluyendo encabezado y pie
		headsep = 1.0cm,          	% Separaci?n entre el encabezado y el texto
		footskip = 1.5cm,        	% Separaci?n entre el texto y el pie
		headheight = 15pt
		}

\else\ifEPUB

% No res

\else % llibre o tesi

	\newlength{\sangratMarques}

	\ifCROP
 		\setlength{\sangratMarques}{3mm}
	\else
 		\setlength{\sangratMarques}{0mm}
	\fi	
		
	\newlength{\amplePublicacio}
	\setlength{\amplePublicacio}{17cm + 2\sangratMarques}
	
	\newlength{\alcadaPublicacio}
	\setlength{\alcadaPublicacio}{24cm + 2\sangratMarques}
	
	\newlength{\margeVerticalSuperior}
	\setlength{\margeVerticalSuperior}{1.75cm + \sangratMarques}

	\newlength{\margeVerticalInferior}
	\setlength{\margeVerticalInferior}{1.75cm + \sangratMarques}
	
	\newlength{\margeInterior}
	\setlength{\margeInterior}{2.5cm + \sangratMarques}

	\geometry{
		twoside,
		vmargin = {\margeVerticalSuperior, \margeVerticalInferior},
		includehead, includefoot,
		left = \margeInterior,
		headheight = 0.5cm,	
		headsep = 0.75cm,
		footskip = 1.25cm,
		textwidth = 13cm, %totalheight = 20cm, % Caja de texto 13cm x 21 cm
		paperwidth = \amplePublicacio, paperheight = \alcadaPublicacio,
		}

	% --------------------------------------------
	% Marques de tall
	
	\ifCROP
		\usepackage[cross, a4, center, odd, noinfo]{crop}
	\fi

\fi\fi\fi


% ---------------------------------------------------------------------
% Capçaleres i peus de pàgina

\RequirePackage{fancyhdr}
\pagestyle{fancy}

\renewcommand{\chaptermark}[1]{\markboth{\@chapapp\ \thechapter.\ #1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}{}}

\fancyhead{} % Esborra la configuraci? de la cap?alera
\fancyfoot{} % Esborra la configuraci? del peu


\newlength{\myevenheadlinehoffset}
\newlength{\myevenheadtexthoffset}
\newlength{\myoddheadlinewidth}
\newlength{\myevenheadlinewidth}

\setlength{\myevenheadlinehoffset}{0.0cm}
\setlength{\myevenheadtexthoffset}{0.0cm}
\setlength{\myoddheadlinewidth}{\textwidth}
\setlength{\myevenheadlinewidth}{\textwidth}

\newlength{\myevenfootlinehoffset}
\newlength{\myoddfootlinehoffset}
\newlength{\myfootlinewidth}

\setlength{\myevenfootlinehoffset}{0cm}
\setlength{\myoddfootlinehoffset}{\textwidth}
\setlength{\myfootlinewidth}{1cm}


\ifEBOOKPDF

	\newlength{\hMargins}
	\setlength{\hMargins}{\paperwidth-\textwidth}
	\newlength{\hMargin}
	\setlength{\hMargin}{0.5\hMargins}
	\newlength{\posPagina}
	\setlength{\posPagina}{\textwidth + 0.5\hMargin}

	\RequirePackage{lastpage}
	\RequirePackage{xcolor}
	\RequirePackage{tikz}
	
	\definecolor{grisUPV}{HTML}{313233}
	\definecolor{grisCLARET}{HTML}{EEEEEE}
	
	\renewcommand{\chaptermark}[1]{\markboth{\@chapapp\ \thechapter.\ #1}{}}

	\fancyhf{} %
	\fancyhead[L]{%
		\begin{tikzpicture}[overlay]
			\draw[fill=grisUPV] (-\hMargin,2ex) rectangle ++(\paperwidth,-1.0cm);
			\node[right] (leftMark) at (-0.75\hMargin,0) {\color{grisCLARET}\sffamily\mdseries\scriptsize\leftmark};
			\node[right] (rightMark) at (-0.5\hMargin,-2ex) {\color{grisCLARET}\sffamily\mdseries\scriptsize\rightmark};
			\node[] (thePage) at (\posPagina,-1ex) {\color{grisCLARET}\sffamily\small\thepage/\pageref*{LastPage}};
		\end{tikzpicture}
		}

	\fancypagestyle{plain}{%
	\fancyhf{} % clear all header and footer fields
	\fancyhead[L]{%
		\begin{tikzpicture}[overlay]
			\draw[fill=grisUPV] (-\hMargin,2ex) rectangle ++(\paperwidth,-1.0cm);
			\node[] (thePage) at (\posPagina,-1ex) {\color{grisCLARET}\sffamily\small\thepage/\pageref*{LastPage}};
		\end{tikzpicture}
		}}
	
	\renewcommand{\headrulewidth}{0.0pt}
	\renewcommand{\footrulewidth}{0.0pt}
	
	\renewcommand{\frontmatter}{}
	\renewcommand{\mainmatter}{}
	
\else\ifAquatre

%	\fancyhead{}
	\fancyhead[LE]{\small\itshape\nouppercase\leftmark}
	\fancyhead[RO]{\small\itshape\nouppercase\rightmark}

%	\fancyfoot{}
	\fancyfoot[LE,RO]{\thepage}

	\renewcommand{\headrulewidth}{0.25pt}
	\renewcommand{\footrulewidth}{0.25pt}

\else\ifEPUB

% Nores

\else % llibre o tesi

	\fancyhead[LE]{%
		\hspace*{\myevenheadlinehoffset}\makebox[0cm][l]{\rule[-2mm]{\myevenheadlinewidth}{.25pt}}%
		\hspace*{\myevenheadtexthoffset}{\footnotesize\itshape\nouppercase\leftmark}%
		}
	\fancyhead[LO]{%
		\makebox[0cm][l]{\rule[-2mm]{\myoddheadlinewidth}{.25pt}}%
		\makebox[\textwidth][r]{\footnotesize\itshape\nouppercase\rightmark}%
		}
		
	\fancyfoot[LE]{%
		\makebox[0cm][l]{\rule[2.5ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][l]{\thepage}}%
		}

	\fancyfoot[RO]{%
		\hspace*{-\myfootlinewidth}\makebox[0cm][l]{\rule[2.5ex]{\myfootlinewidth}{.25pt}}%
		{\makebox[\myfootlinewidth][r]{\thepage}}%
		}

	% Peu de la primera pàgina del capítol
		
	\fancypagestyle{plain}{%
	    \fancyhf{} % clear all header and footer fields
		\fancyfoot[RO]{%
			\hspace*{-\myfootlinewidth}\makebox[0cm][l]{\rule[2.5ex]{\myfootlinewidth}{.25pt}}%
			{\makebox[\myfootlinewidth][r]{\thepage}}%
			}
		}

	\renewcommand{\headrulewidth}{0.0pt}
	\renewcommand{\footrulewidth}{0.0pt}

\fi\fi\fi



% ---------------------------------------------------------------------
% Format de paràgraf y maquetació

\ifAquatre

	\setlength{\parskip}{3ex}
	\setlength{\parindent}{0pt}

\else

	\setlength{\parskip}{2ex}
	\setlength{\parindent}{0pt}

\fi

\linespread{1.0}

\setlength{\widowpenalty}{10000pt}
\setlength{\clubpenalty}{10000pt}

\raggedbottom


% ---------------------------------------------------------------------
% Mojora los t?tulos de las figuras y tablas

\RequirePackage{caption}
\renewcommand{\captionlabelfont}{\bfseries\small}
\renewcommand{\captionfont}{\small}

% ------------------------------------------------------------------------
% Formato de las secciones

\RequirePackage[
	raggedright,
	compact,
	nobottomtitles*, % Evita que queden t?tols solts al final de la p?gina
	clearempty, % Modifica \cleardoublepage per a que els fulls parells buides siguen blanques
	]{titlesec}

% ------------------------------------------------------------------------

\titleformat{\part}
	[display]
	{\thispagestyle{empty}\normalfont\bfseries\filcenter
	\tolerance=10000\hyphenpenalty=10000}
	{\fontsize{24}{26}\partname\enspace\thepart}
	{1pc}
	{\fontsize{24}{26}}

% ------------------------------------------------------------------------

\titleformat{\chapter}
	[display]
	{\normalfont\Large\filcenter}
	{\raggedleft\fontsize{18}{20}\bfseries\chaptertitlename\enspace\thechapter}
	{2ex}
	{\raggedleft\fontsize{24}{30}\bfseries}%\usefont{T1}{cmss}{n}{n}\selectfont
	[\vspace{2cm}]


% ------------------------------------------------------------------------

\ifAquatre

	% ------------------------------------------------------------------------
	
	\titleformat{\section}
		[hang]
		{\vspace{2ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\Large\bfseries\thesection}
		{1em}
		{\Large\bfseries}
	
	% ------------------------------------------------------------------------
	
	\titleformat{\subsection}
		[hang]
		{\vspace{1.5ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\large\bfseries\itshape\thesubsection}
		{1em}
		{\large\bfseries\itshape}
		[\vspace{-1ex}]
	

\else\ifEBOOKPDF


	% ------------------------------------------------------------------------
	
	\titleformat{\section}
		[hang]
		{\vspace{2ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\fontsize{13}{15}\bfseries\thesection}
		{1em}
		{\fontsize{15}{15}\bfseries}
	
	% ------------------------------------------------------------------------
	
	\titleformat{\subsection}
		[hang]
		{\vspace{1.5ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\fontsize{12}{14}\bfseries\itshape\thesubsection}
		{1em}
		{\fontsize{12}{14}\bfseries\itshape}
		[\vspace{-1ex}]


\else % Llibre o tesi


	% ------------------------------------------------------------------------
	
	\titleformat{\section}
		[hang]
		{\vspace{2ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\fontsize{12}{14}\bfseries\thesection}
		{1em}
		{\fontsize{12}{14}\bfseries}
	
	% ------------------------------------------------------------------------
	
	\titleformat{\subsection}
		[hang]
		{\vspace{1.5ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
		{\fontsize{10.5}{12.5}\bfseries\itshape\thesubsection}
		{1em}
		{\fontsize{10.5}{12.5}\bfseries\itshape}
		[\vspace{-1ex}]
	
\fi\fi


% ------------------------------------------------------------------------

\titleformat{\subsubsection}
	[hang]
	{\vspace{2ex}\raggedright\tolerance=10000\hyphenpenalty=10000}
	{\normalsize\itshape\thesubsubsection}
	{1em}
	{\normalsize\itshape}
	[\vspace{-0.75ex}]

% -------------------------------------------------------
% Para controlar la numeraci?n y formato de la tabla de contenidos

\setcounter{tocdepth}{1}

\RequirePackage{titletoc}

\titlecontents{part}
	[0em]
	{\addvspace{4ex}\Large}
	{\partname\enspace\thecontentslabel\enspace}
	{}
	{\hfill\contentspage}
	[\vspace{-1ex}]
	
\titlecontents{chapter}
	[0em]
	{\addvspace{3ex}\mdseries\large}
	{\thecontentslabel\enspace}
	{}
	{\hfill\contentspage}
	[\vspace{-1ex}]
	
\titlecontents{section}
	[1.5em]
	{\addvspace{.5ex}\small}
	{\thecontentslabel\enspace}
	{}
	{\titlerule*[0.5pc]{.}\contentspage}
	[\vspace{-1.5ex}]	
	
\titlecontents{subsection}
	[3.5em]
	{\vspace{.25ex}\footnotesize}
	{\thecontentslabel\enspace}
	{}
	{\titlerule*[0.5pc]{.}\contentspage}
	[\vspace{-1.5ex}]


\ifAquatre
\titlecontents{section}
	[1.5em]
	{\addvspace{0ex}\small}
	{\thecontentslabel\enspace}
	{}
	{\titlerule*[0.5pc]{.}\contentspage}
	[\vspace{-1.5ex}]	
\fi

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------


\ifEPUB
	\setlength{\partopsep}{-1ex plus 1ex minus 0ex} % Configurat per enumitem
	
	% ------------------------------------------------------
	% Compatibilitat tex4ht - biblatex
	% Gener de 2013
	
	\makeatletter
	\ifdefined\blx@unitmark
	\else
		\newcommand\blx@unitmark{23sp}
	\fi
	\makeatother		

\else
	\RequirePackage{enumitem}
	\setlist[1]{partopsep=-1ex,parsep=\parskip,itemsep=0\parskip}
	\setlist[2]{partopsep=-1ex,parsep=\parskip,itemsep=0\parskip}
\fi


\makeatletter
\let\ifes@LaTeXe\iftrue % Per a la compatibilitat amb babel des de htlatex
\makeatother

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------

\RequirePackage{mathtools}

\ifDistanciesEquacions
\AtBeginDocument
	{
	% ---------------------------------------------------
	% Dist?ncies de les equacions al text
	
	% Per a les equacions normals
	\abovedisplayshortskip = -1.0ex plus 0ex minus 0.25ex
	\belowdisplayshortskip = 2.0ex plus 1ex minus 0.0ex
	
	% Per a les equacions en varies l?nies
	\abovedisplayskip = -1.0ex plus 0ex minus 0.25ex
	\belowdisplayskip = 2.0ex plus 1ex minus 0.0ex
	}
\fi

% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
